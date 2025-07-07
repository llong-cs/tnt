'''
model that supports cross-table generation
'''
from model.qformer_projector import Projector
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoConfig, PretrainedConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Sequence, List, Tuple, Union
import json, os
import torch
from model.utils import find_correct_case_file_name, tokenize_insert
from torch import nn
import numpy as np
from model.utils import *
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from deepspeed import get_accelerator
from Table_Encoder.model.load_encoder import load_encoder
from safetensors.torch import load_file
from config import SENTENCE_TRANSFORMER_PATH, MAX_ROW, MAX_COL, IGNORE_INDEX, ENCODER_PATH, MISTRAL_MODEL_PATH
if 'MAX_COL' in os.environ:
    MAX_COL = int(os.environ['MAX_COL'])
    print(f'find new MAX_COL in environ: {MAX_COL}')
if 'MAX_ROW' in os.environ:
    MAX_ROW = int(os.environ['MAX_ROW'])
    print(f'find new MAX_ROW in environ: {MAX_ROW}')
    
class Model(nn.Module):

    def __init__(self, *, load_pretrained = True):
        super().__init__()
        self.decoder = AutoModelForCausalLM.from_pretrained(MISTRAL_MODEL_PATH, torch_dtype = torch.bfloat16, attn_implementation="flash_attention_2").to('cuda')

        self.encoder = load_encoder(path = (ENCODER_PATH if load_pretrained else None)).to(dtype = torch.bfloat16).to('cuda')
        self.projector = Projector(
            encoder_hidden_size=384,
            decoder_hidden_size=4096,
            dim_head=128,
            num_queries=5,
        ).to(dtype = torch.bfloat16)
        self.encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_PATH)
        self.gradient_checkpointing_enable = self.decoder.gradient_checkpointing_enable
        
        
    @classmethod
    def from_pretrained(cls, path):
        print('loading pretrained model...')
        model = cls(load_pretrained = False)
        print('model initialized')

        model.load_state_dict(load_file(os.path.join(path, 'model.safetensors'))) 
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        print('model loaded')
        return model

    def prepare_insert_embeds(
        self, *, input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None, table_embeds, learnable_embeds = None
    ):
        assert learnable_embeds == None, "learnable embeddings is not yet supported"
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_insert_embs = (cur_input_ids == INSERT_EMBS_TOKEN_ID).sum()
            if num_insert_embs == 0:
                raise ValueError("No insert embs token found in the input_ids")
            cur_table_embeds = table_embeds[batch_idx].clone()
            
            insert_emb_token_indices = [-1] + torch.where(cur_input_ids == INSERT_EMBS_TOKEN_ID)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(insert_emb_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[insert_emb_token_indices[i]+1:insert_emb_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[insert_emb_token_indices[i]+1:insert_emb_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.decoder.get_input_embeddings()((torch.cat(cur_input_ids_noim)))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_insert_embs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_insert_embs:
                    assert cur_table_embeds.shape[0] >= i, f"not match: {cur_table_embeds.shape[0]}, {i}"
                    cur_insert_emb_features = cur_table_embeds[i] # num_heads * decode_hidden
                    cur_new_input_embeds.append(cur_insert_emb_features)
                    cur_new_labels.append(torch.full((cur_insert_emb_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
            device = self.decoder.device
            cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)


        if hasattr(self, 'max_length'):
            new_input_embeds = [x[:self.max_length] for x in new_input_embeds]
            new_labels = [x[:self.max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.decoder.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            # new_labels = _labels

        if _attention_mask is None:
            pass # keep the newly created attention mask
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def get_embedded_table(self, path_csv):
        def process_table_df(table_df):
            numeric_columns = table_df.select_dtypes(include=["number"]).columns
            numeric_indices = [
                table_df.columns.get_loc(col) for col in numeric_columns
            ]
            
            # fill missing values with mean
            table_df[numeric_columns] = table_df[numeric_columns].apply(
                lambda col: col.fillna(col.mean() if not col.isna().all() else 0)
            )
            if len(table_df) > MAX_ROW:
                table_df = table_df.sample(n=MAX_ROW)
                
            
            table_np = table_df.to_numpy().astype(str)
            
            return table_np
        def load_tokenized_table(anchor_table):
            anchor_table = process_table_df(anchor_table)
            num_rows, num_cols = anchor_table.shape[0], anchor_table.shape[1]
            anchor_row_num = anchor_table.shape[0]
            anchor_table = anchor_table.reshape(-1)
            max_length = 64
            tokenized_anchor_table = self.encoder_tokenizer(anchor_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')                
            tokenized_anchor_table = {k: v.reshape(anchor_row_num, num_cols, -1) for k, v in tokenized_anchor_table.items()}
            return tokenized_anchor_table

        # print(f'loading csv from {path_csv}')
        table_df = pd.read_csv(
            path_csv,
            encoding="utf-8",
            low_memory=False,
            nrows=500
        )
        df_col_count = table_df.shape[1]
        df_row_count = min(table_df.shape[0], MAX_ROW)
        anchor_table = load_tokenized_table(table_df)
        num_cols = anchor_table['input_ids'].shape[1]
        anchor_table_row_num = anchor_table['input_ids'].shape[0]
        anchor_table_padded = {k: F.pad(v, (0, 0, 0, MAX_COL - v.shape[1], 0, MAX_ROW - v.shape[0]), "constant", 1) for k, v in anchor_table.items()}
        # print('..', anchor_table_padded['input_ids'].shape, anchor_table_padded['attention_mask'].shape, anchor_table_padded['token_type_ids'].shape)
        anchor_table_mask = np.zeros((MAX_ROW, MAX_COL))
        anchor_table_mask[:anchor_table_row_num, : num_cols] = 1
        ret = (
            anchor_table_padded['input_ids'].to(device = self.decoder.device),
            anchor_table_padded['attention_mask'].to(device = self.decoder.device),
            anchor_table_padded['token_type_ids'].to(device = self.decoder.device),
            torch.tensor(anchor_table_mask, device = self.decoder.device),
            df_col_count,
            df_row_count
        )
        return ret
    
    def get_encoder_output(self, path_csv):
        # path_csv: list of list of csv paths, from all batches
        
        table_count = [len(c_list) for c_list in path_csv]
        column_count = []
        cat_table_embeds = [[] for _ in range(len(table_count))] # len(table_count) = batch_size
        cat_key_padding_mask = [[] for _ in range(len(table_count))]
        
        for batch_idx, c_list in enumerate(path_csv):
            anchor_table_input_ids = []
            anchor_table_attention_mask = []
            anchor_table_token_type_ids = []
            anchor_table_mask = []
            cur_column_count = []
            cur_row_count = []
            for c in c_list:
                p, q, r, s, col_cnt, row_cnt = self.get_embedded_table(c)
                cur_column_count.append(col_cnt)
                cur_row_count.append(row_cnt)
                anchor_table_input_ids.append(p)
                anchor_table_attention_mask.append(q)
                anchor_table_token_type_ids.append(r)
                anchor_table_mask.append(s)
                
            column_count.append(cur_column_count) 
            
            anchor_table_input_ids = torch.stack(anchor_table_input_ids, dim=0).to(self.decoder.device)
            anchor_table_attention_mask = torch.stack(anchor_table_attention_mask, dim=0).to(self.decoder.device)
            anchor_table_token_type_ids = torch.stack(anchor_table_token_type_ids, dim=0).to(self.decoder.device)
            anchor_table_mask = torch.stack(anchor_table_mask, dim=0).to(self.decoder.device)
            encoder_output = self.encoder(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask, inference=True) # shape: (table_count, num_rows, num_cols, encoder_dim)
            encoder_output = encoder_output.permute(0, 2, 1, 3) # shape: (table_count, MAX_COL, MAX_ROW, encoder_dim)
            for table_idx, (col, row) in enumerate(zip(cur_column_count, cur_row_count)):
                assert encoder_output[table_idx].shape[0] >= col, f"not match: {encoder_output[table_idx].shape[0]}, {col}"
                cat_table_embeds[batch_idx].append(encoder_output[table_idx, :col]) # (col, MAX_ROW, encoder_dim)
                # padding mask should be: (num_cols, num_rows)
                cur_mask = torch.zeros(col, MAX_ROW, dtype=torch.bool, device=self.decoder.device)
                cur_mask[:, row:] = True
                cat_key_padding_mask[batch_idx].append(cur_mask)
            cat_table_embeds[batch_idx] = torch.cat(cat_table_embeds[batch_idx], dim = 0) # shape: (num_cols, MAX_ROW, encoder_dim)
            cat_key_padding_mask[batch_idx] = torch.cat(cat_key_padding_mask[batch_idx], dim = 0) # shape: (num_cols, MAX_ROW)

        return cat_table_embeds, cat_key_padding_mask

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        path_csv: Optional[str] = None,
        insert_embs = None  
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        table_embeds, key_padding_mask = self.get_encoder_output(path_csv)
        for i, (embeds, mask) in enumerate(zip(table_embeds, key_padding_mask)):
            table_embeds[i] = self.projector(embeds, mask)
        # print(table_embeds[0].shape)    
        
        try:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_insert_embeds(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                table_embeds=table_embeds,
            )
        except Exception as e:
            raise e
        # print(inputs_embeds)
        ret = self.decoder.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.to(dtype = self.decoder.dtype),
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return ret
    @torch.inference_mode()
    def generate(self, input_str: List = None, path_csv: List = None, max_new_tokens = 1024, half = False, **kwargs):
        bs = len(input_str)
        if '<insert_embs>' in input_str[0]:   
            table_embeds, key_padding_mask = self.get_encoder_output(path_csv)
            for i, (embeds, mask) in enumerate(zip(table_embeds, key_padding_mask)):
                table_embeds[i] = self.projector(embeds, mask)
            if half:
                return table_embeds
        
        
        inputs_embeds = []
        attention_mask = []
        for i in range(bs):
            if '<insert_embs>' in input_str[i]:
                cur_input_ids = tokenize_insert_llama3(input_str[i], self.tokenizer).unsqueeze(0).to(device = self.decoder.device)
                cur_table_embeds = table_embeds[i].unsqueeze(0)

                (
                    input_ids,
                    position_ids,
                    cur_attention_mask,
                    past_key_values,
                    cur_inputs_embeds,
                    labels,
                ) = self.prepare_insert_embeds(
                    input_ids=cur_input_ids,
                    # position_ids,
                    table_embeds=cur_table_embeds,
                )
                
                inputs_embeds.append(cur_inputs_embeds)
                attention_mask.append(torch.ones(cur_inputs_embeds.shape[:-1], device=self.decoder.device, dtype = torch.int64))
            else:
                raise NotImplementedError
            
        longest_input = max([x.shape[1] for x in inputs_embeds])
        inputs_embeds_padded = torch.zeros(bs, longest_input, *inputs_embeds[0].shape[2:], device = self.decoder.device, dtype = inputs_embeds[0].dtype)
        attention_mask_padded = torch.zeros(bs, longest_input, device = self.decoder.device, dtype = torch.int64)
        for i in range(bs):
            inputs_embeds_padded[i, longest_input - inputs_embeds[i].shape[1]:] = inputs_embeds[i][0]
            attention_mask_padded[i, longest_input - inputs_embeds[i].shape[1]:] = attention_mask[i][0]

        inputs_embeds = inputs_embeds_padded
        attention_mask = attention_mask_padded
        
        
        
        ret = self.decoder.generate(
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.to(dtype = self.decoder.dtype),
            # eos_token_id = self.tokenizer.eos_token_id,
            use_cache=True,
            **kwargs
        )
        
        ret_str = []
        for i in range(bs):
            stop_sign = self.tokenizer.eos_token_id
            # stop_sign = 151645
            ret_list = ret[i].tolist()
            if stop_sign in ret_list:
                if stop_sign in ret_list:
                    ret_list = ret_list[:ret_list.index(stop_sign)]
            ret_str.append(self.tokenizer.decode(ret_list))
        print(ret_str)
        return ret, ret_str