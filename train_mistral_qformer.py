import copy
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)

from datasets import load_dataset
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
from torch import nn
import transformers
import tokenizers
import numpy as np
from torch.utils.data import Dataset
from Table_Encoder.model.load_encoder import load_encoder
from model.model_sft_qf_mistral import Model
from model.utils import build_instruction

from model.utils import find_correct_case_file_name, build_plain_instruction_prompt, tokenize_insert_llama3

from config import INSERT_EMBS_TOKEN, INSERT_EMBS_TOKEN_ID, SENTENCE_TRANSFORMER_PATH, INSERT_START_TOKEN, INSERT_END_TOKEN, MISTRAL_MODEL_PATH

IGNORE_INDEX = -100
 

# EOT_TOKEN = "<|EOT|>"

@dataclass
class ModelArguments:
    load_pretrained: bool = field(default=False)
    pretrained_path: str = field(default=None)
    encoder_path: str = field(default=None, metadata={"help": "Path of pretrained encoder"})
    projector_path: str = field(default = None)
    device: str = field(default="cuda")
    
    
    encoder_hidden_size: int = field(default=None)
    decoder_hidden_size: int = field(default=None)
    torch_dtype: str = field(default="float32")
    

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_count: int = field(default=1000000)
    eval_data_path : str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    gradient_checkpointing: bool = field(default=False),
    cache_dir: Optional[str] = field(default='xxx')    
    weight_decay: float = field(default=0.01)

    freeze_projector: bool = field(default=False)
    freeze_encoder: bool = field(default=False)
    freeze_decoder: bool = field(default=True)
    model_max_length: int = field(default=4096)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



    
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, insert_embs = False) -> Dict:
    """Tokenize a list of strings."""
    if not insert_embs:
        tokenized_list = [
            tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",  
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for prompt in strings
        ]

        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
    else:
        tokenized_list = [
            tokenize_insert_llama3(prompt, tokenizer)
            for prompt in strings
        ]
        input_ids = labels = tokenized_list
        input_ids_lens = labels_lens = [
            tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    insert_embs: bool = False
) -> Dict:
    """
    Preprocess the data by tokenizing.
    Parameters:
        - sources: instructions
        - targets: outputs
    Returns:
        - input_ids: tokenized instructions
        - labels: tokenized outputs, padded with IGNORE_INDEX of the same length as the input_ids
    """

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, insert_embs=insert_embs) for strings in (examples, sources)] 
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        path_csv = [instance['path_csv'] for instance in instances]
        
        insert_embs = [instance.get('insert_embs', False) for instance in instances]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            path_csv=path_csv,
            insert_embs = insert_embs
        )

def train_tokenize_function(examples, tokenizer):
    
    
    sources = [
        build_instruction(cur_q, tokenizer=tokenizer)
        for cur_q in examples['instruction']
    ]
    
    EOS_TOKEN = tokenizer.eos_token

    targets = [f"{output}\n{EOS_TOKEN}" for output in examples['answer']]

    is_insert = True
    data_dict = preprocess(sources, targets, tokenizer, insert_embs=is_insert)
    data_dict['path_csv'] = examples['path_csvs']
        
    data_dict['insert_embs'] = [is_insert] * len(data_dict['input_ids'])
    return data_dict


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.torch_dtype = eval('torch.' + model_args.torch_dtype)

        
    if training_args.local_rank == 0:
        print('='*100)

    print('loading tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MISTRAL_MODEL_PATH,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True
    )
    print('tokenizer loaded')
    
    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<unk>'
        print("    ----->", tokenizer.pad_token, tokenizer.pad_token_id)
        
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
    
    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(MISTRAL_MODEL_PATH))
    if model_args.load_pretrained == True:
        print('load pretrained model')
        model = Model.from_pretrained(model_args.pretrained_path).to('cuda', dtype = torch.bfloat16)
    else:
        model = Model().to('cuda', dtype = torch.bfloat16)

    model.tokenizer = tokenizer
    model.max_length = training_args.model_max_length
    print('max length: ', model.max_length)
    if training_args.freeze_decoder:
        model.decoder.requires_grad_(False)
        model.decoder.eval()
        print('freeze decoder')
    else:
        model.decoder.requires_grad_(True)
        model.decoder.train()
        

    if training_args.freeze_encoder:
        model.encoder.eval()
        model.encoder.requires_grad_(False)
        
    else:
        model.encoder.requires_grad_(True)
        model.encoder.train()
        
    if training_args.freeze_projector:
        model.projector.requires_grad_(False)
    else:
        model.projector.requires_grad_(True)
        
    for param in model.decoder.parameters():
        assert param.requires_grad != training_args.freeze_decoder
    
    model.projector.max_length = training_args.model_max_length
    if training_args.local_rank == 0:
        print('model', model)

    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        cache_dir='xxx',
        split='train'
    )
    raw_eval_datasets = load_dataset(
        'json',
        data_files=data_args.eval_data_path,
        cache_dir='xxx',
        split="train",
    )
    
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=512,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )
    eval_dataset = raw_eval_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=512,
        remove_columns=raw_eval_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )
    

    if training_args.local_rank == 0 and training_args.world_size > 1:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 1):
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']},\n {train_dataset[index]['labels']}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    trainer = Trainer(model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)
    trainer.eval_data_path = data_args.eval_data_path

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.local_rank == 0:
        try:
            projector = model.projector.cpu()
            torch.save(projector, os.path.join(training_args.output_dir, 'projector.bin'))
            print('save projector')
        except Exception as e:
            print('fail to save projector', e)
        
        try:
            encoder = model.encoder.cpu()
            torch.save(encoder, os.path.join(training_args.output_dir, 'encoder.bin'))
            print('save encoder')
        except Exception as e:
            print('fail to save encoder', e)
        
        try:
            if training_args.freeze_decoder == False:
                decoder = model.decoder.cpu()
                torch.save(decoder, os.path.join(training_args.output_dir, 'decoder.bin'))
                print('save decoder')
        except:
            print('fail to save decoder')
            
if __name__ == "__main__":
    import wandb
    train()
