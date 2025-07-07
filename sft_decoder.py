import copy
import random, tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import Trainer
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
from model.utils import build_instruction

from model.utils import find_correct_case_file_name, build_plain_instruction_prompt, tokenize_insert
from config import INSERT_EMBS_TOKEN, INSERT_EMBS_TOKEN_ID, INSERT_SEP_TOKEN, SENTENCE_TRANSFORMER_PATH

IGNORE_INDEX = -100
 

# EOT_TOKEN = "<|EOT|>"

@dataclass
class ModelArguments:
    decoder_path: str = field(default=None, metadata={"help": "Path of pretrained decoder"})
    

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path : str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    gradient_checkpointing: bool = field(default=False),
    cache_dir: Optional[str] = field(default='xxx')
    optim: str = field(default="adamw_torch")
    weight_decay: float = field(default=0.1)
    model_max_length: int = field(default=4096)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



    
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
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
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)] 
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
        
        # path_csv = [instance['path'] for instance in instances]
        

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    
    sources = [
        build_instruction(cur_q.replace('<insert><insert_embs></insert>|', ''), tokenizer=tokenizer)
        for cur_q in examples['instruction']
    ]
    
    EOT_TOKEN = tokenizer.eos_token
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['answer']]

    data_dict = preprocess(sources, targets, tokenizer)
        
    return data_dict


    
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.torch_dtype = torch.bfloat16

        
    if training_args.local_rank == 0:
        print('='*100)
        # print(training_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.decoder_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<unk>'
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
 
    
    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.decoder_path))
    
    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.decoder_path, torch_dtype = torch.bfloat16, attn_implementation="flash_attention_2").to(device = training_args.device) #
    print(model.dtype)
        
    model.train()
    model.requires_grad_(True)

    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        cache_dir='xxx',
        split="train",
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
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )
    
    

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {len(train_dataset[index]['input_ids'])}, {train_dataset[index]['labels']}.")
            # print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    trainer = Trainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)



    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

            
if __name__ == "__main__":
    import wandb
     
    train()
    