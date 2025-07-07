import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import os
import math
import random
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
DATASETS_PATH = os.environ["DATASETS_PATH"]
MODELS_PATH = os.environ["MODELS_PATH"]
SENTENCE_TRANSFORMER = os.environ["SENTENCE_TRANSFORMER"]

MAX_COL=
MAX_ROW=

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def list_npy_csv_files(data_path, from_csv=False):
    if not from_csv:
        file_list = os.listdir(data_path)
        npy_files = []
        for n_file in file_list:
            if n_file.endswith(".npy"):
                npy_files.append(os.path.join(data_path, n_file))
        return npy_files
    else:
        # for 2-level directory structure
        # file_list = os.listdir(data_path)
        # csv_files = []
        # for d_file in file_list:
        #     csv_file_list = os.listdir(os.path.join(data_path, d_file))
        #     for c_file in csv_file_list:
        #         if c_file.endswith(".csv"):
        #             csv_files.append(os.path.join(data_path, d_file, c_file))
        # return csv_files
        
        # for 1-level directory structure
        file_list = os.listdir(data_path)
        csv_files = []
        for c_file in file_list:
            if c_file.endswith(".csv"):
                csv_files.append(os.path.join(data_path, c_file))
        return csv_files


def list_2npy_files(data_path):
    file_list = os.listdir(data_path)
    npy_files = []
    for file in file_list:
        if file.endswith(".npy") or file.endswith(".csv"):
            npy_files.append(file)
    res = []
    for i in range(0, len(npy_files)):
        for j in range(i, len(npy_files)):
            res.append([npy_files[i], npy_files[j]])
            if len(res) >= 3000:
                break
        if len(res) >= 3000:
            break
    return res


def list_same_col_files(data_path):
    res = []
    col_dir_list = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]
    for col_dir in col_dir_list:
        file_list = os.listdir(os.path.join(data_path, col_dir))
        npy_files = []
        for file in file_list:
            if file.endswith(".npy") or file.endswith(".csv"):
                npy_files.append(os.path.join(data_path, col_dir, file))
        neg_num = 0
        for i in range(len(npy_files)):
            for j in range(i + 1, len(npy_files)):
                res.append([npy_files[i], npy_files[j]])
                neg_num += 1
                if neg_num >= 20:
                    break
            if neg_num >= 20:
                break
        for k in range(neg_num):
            rand = random.randint(0, len(npy_files) - 1)
            res.append([npy_files[rand], npy_files[rand]])
    return res


def generate_special_tokens():
    model = SentenceTransformer(f"{MODELS_PATH}/{SENTENCE_TRANSFORMER}")
    embedding_dim = model.get_sentence_embedding_dimension()
    shuffle_tokens_num = 100
    shuffle_tokens = model.encode(
        [f"[unused{i}]" for i in range(shuffle_tokens_num)]
    )  # [unused1]<col1>[unused1] 50*(100+2st)*384?
    cls_token = model.encode("[CLS]")
    sep_token = model.encode("[SEP]")
    np.savez(
        f"data/special_tokens_{SENTENCE_TRANSFORMER}.npz",
        shuffle_tokens=shuffle_tokens,
        cls_token=cls_token,
        sep_token=sep_token,
    )


def get_special_tokens():
    special_tokens = np.load(f"data/special_tokens_{SENTENCE_TRANSFORMER}.npz")
    return (
        torch.from_numpy(special_tokens["cls_token"]),
        torch.from_numpy(special_tokens["sep_token"]),
        torch.from_numpy(special_tokens["shuffle_tokens"]),
    )


def get_device(module):
    if next(module.parameters(), None) is not None:
        return next(module.parameters()).device
    elif next(module.buffers(), None) is not None:
        return next(module.buffers()).device
    else:
        raise ValueError("The module has no parameters or buffers.")

def is_convertible_to_numeric(series):
    try:
        pd.to_numeric(series)
        return True
    except ValueError:
        return False

class TableDataset(Dataset):
    def __init__(
        self,
        data_path,
        pred_type="contrastive",
        from_csv=False,
        model=None,
        idx=None,
        shuffle_num=3,
        numeric_mlp=False
    ):
        self.data_path = data_path
        self.pred_type = pred_type
        self.numeric_mlp = numeric_mlp
                
        # load the learned speicial tokens from a certain SentenceTransformer
        # self.cls_token, self.sep_token, self.shuffle_tokens = get_special_tokens()
        
        self.from_csv = from_csv
        # self.model = model
        self.shuffle_num = (
            3  # num of cols that need to be shuffled, won't influence contrastive
        )
        
        # read data list
        self.table_list = np.load(self.data_path)
        
        self.max_row = MAX_ROW
        self.max_col = MAX_COL
        
        # scan data file
        # self.table_list = list_npy_csv_files(self.data_path, self.from_csv)
        # self.table_list = np.array(self.table_list)
            
        if idx is not None:
            self.table_list = self.table_list[idx]
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODELS_PATH}/{SENTENCE_TRANSFORMER}", use_fast=False)
        self.st_name = SENTENCE_TRANSFORMER

    def __len__(self):
        return len(self.table_list)
    
    def process_table_df(self, table_df):
        if len(table_df.columns) > self.max_col:
            table_df = table_df.sample(n=self.max_col, axis=1)
        
        numeric_columns = table_df.select_dtypes(include=["number"]).columns
        numeric_indices = [
            table_df.columns.get_loc(col) for col in numeric_columns
        ]
        
        # fill missing values with mean
        table_df[numeric_columns] = table_df[numeric_columns].apply(
            lambda col: col.fillna(col.mean() if not col.isna().all() else 0)
        )
        
        if len(table_df) > self.max_row * 2:
            table_df = table_df.sample(n=self.max_row * 2)
        
        table_np = table_df.to_numpy().astype(str)
        
        return table_np
    
    def load_tokenized_table(self, table_file):
        tokenizer = self.tokenizer
        
        table_df = pd.read_csv(
            table_file,
            encoding="utf-8",
            low_memory=False,
            nrows=500
        )

        # size = [num_rows, num_cols]
        table = self.process_table_df(table_df)
        num_rows, num_cols = table.shape[0], table.shape[1]
        
        anchor_table, shuffled_table = self.split_table(table)
        
        anchor_row_num = anchor_table.shape[0]
        shuffled_row_num = shuffled_table.shape[0]
                
        shuffled_table, shuffle_idx = self.shuffle_table(shuffled_table)
        
        anchor_table, shuffled_table = anchor_table.reshape(-1), shuffled_table.reshape(-1)

        # size = [num_cells, seq_len]
        # 
        
        max_length = 64
        tokenized_anchor_table = tokenizer(anchor_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        tokenized_shuffled_table = tokenizer(shuffled_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
                
        tokenized_anchor_table = {k: v.reshape(anchor_row_num, num_cols, -1) for k, v in tokenized_anchor_table.items()}
        tokenized_shuffled_table = {k: v.reshape(shuffled_row_num, num_cols, -1) for k, v in tokenized_shuffled_table.items()}
                
        assert tokenized_anchor_table['input_ids'].shape[2] == tokenized_shuffled_table['input_ids'].shape[2]
        
        return tokenized_anchor_table, tokenized_shuffled_table, shuffle_idx

    def split_table(self, table):
        num_rows = table.shape[0]
        anchor_table_row_num = num_rows // 2
        shuffled_table_row_num = num_rows - anchor_table_row_num
        
        anchor_table = table[:anchor_table_row_num]
        shuffled_table = table[-shuffled_table_row_num:]
        
        return anchor_table, shuffled_table
    
    def shuffle_table(self, shuffled_table):
        # Shuffle columns
        # Randomly select columns to shuffle
        shuffle_idx = torch.randperm(shuffled_table.shape[1])
        shuffled_table = shuffled_table[:, shuffle_idx]
        
        return shuffled_table, shuffle_idx

    def __getitem__(self, idx):
        anchor_table, shuffled_table, shuffle_idx = self.load_tokenized_table(self.table_list[idx])
        num_cols = anchor_table['input_ids'].shape[1]
        
        anchor_table_row_num = anchor_table['input_ids'].shape[0]
        shuffled_table_row_num = shuffled_table['input_ids'].shape[0]
            
        anchor_table_padded = {k: F.pad(v, (0, 0, 0, self.max_col - v.shape[1], 0, self.max_row - v.shape[0]), "constant", 1) for k, v in anchor_table.items()}
        shuffled_table_padded = {k: F.pad(v, (0, 0, 0, self.max_col - v.shape[1], 0, self.max_row - v.shape[0]), "constant", 1) for k, v in shuffled_table.items()}

        anchor_table_mask = np.zeros((self.max_row, self.max_col))
        shuffled_table_mask = np.zeros((self.max_row, self.max_col))

        anchor_table_mask[:anchor_table_row_num, : num_cols] = 1
        shuffled_table_mask[:shuffled_table_row_num, : num_cols] = 1

        shuffle_idx_padded = F.pad(shuffle_idx, (0, self.max_col - len(shuffle_idx)), "constant", -1)
                
        if self.st_name == 'all-MiniLM-L6-v2' or self.st_name == 'bge-small-en-v1.5':        
            return (
                anchor_table_padded['input_ids'],
                anchor_table_padded['attention_mask'],
                anchor_table_padded['token_type_ids'],
                shuffled_table_padded['input_ids'],
                shuffled_table_padded['attention_mask'],
                shuffled_table_padded['token_type_ids'],
                anchor_table_mask,
                shuffled_table_mask,
                shuffle_idx_padded
            )
        elif self.st_name == 'puff-base-v1':
            return (
                anchor_table_padded['input_ids'],
                anchor_table_padded['attention_mask'],
                torch.zeros_like(anchor_table_padded['input_ids']),
                shuffled_table_padded['input_ids'],
                shuffled_table_padded['attention_mask'],
                torch.zeros_like(anchor_table_padded['input_ids']),
                anchor_table_mask,
                shuffled_table_mask,
                shuffle_idx_padded
            )
    
            
    # load table embeddings either from raw csv or pre-processed npy files
    def load_table_embeddings(self, table_file, add_col_emb=False):
        if self.from_csv:
            embedding_dim = self.embedding_dim
            tokenizer = AutoTokenizer.from_pretrained(f"{MODELS_PATH}/{SENTENCE_TRANSFORMER}")
                            
            table_df = pd.read_csv(
                table_file,
                encoding="utf-8",
                low_memory=False,
                nrows=100
            )

            table = self.process_table_df(table_df)
            
            table_emb = torch.zeros((table.shape[0], table.shape[1], embedding_dim)).to(device=get_device(self.model))
            for j, row in enumerate(table):
                if self.numeric_mlp:
                    row_emb = torch.zeros((table.shape[1], embedding_dim)).to(device=get_device(self.model))
                    if len(numeric_indices) > 0:
                        row_emb[numeric_indices] = (
                            self.model.module.num_mlp(
                                torch.tensor(
                                    row[numeric_indices]
                                    .astype(np.float32)
                                    .reshape(-1, 1)
                                ).to(device=get_device(self.model))
                            )
                        )
                    if len(non_numeric_indices) > 0:
                        # bge-small-en
                        # encoded_input = tokenizer(row, padding=True, truncation=True, return_tensors='pt')
                        # encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                        # row_emb = self.model.module.st(**encoded_input) # for multi-gpu
                        # table_emb[j] = F.normalize(row_emb[0][:, 0], p=2, dim=1)
                        
                            # all-minilm
                        encoded_input = tokenizer(row[non_numeric_indices].astype(str).tolist(), padding=True, truncation=True, return_tensors='pt')
                        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                        encoded_output = self.model.module.st(**encoded_input) # for multi-gpu
                        encoded_output = mean_pooling(encoded_output, encoded_input['attention_mask'])
                        row_emb[non_numeric_indices] = F.normalize(encoded_output, p=2, dim=1)
                    table_emb[j] = row_emb
                else:
                    encoded_input = tokenizer(row.astype(str).tolist(), padding=True, truncation=True, return_tensors='pt')
                    encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                    encoded_output = self.model.module.st(**encoded_input) # for multi-gpu
                    encoded_output = mean_pooling(encoded_output, encoded_input['attention_mask'])
                    table_emb[j] = F.normalize(encoded_output, p=2, dim=1)
                
            column_names = table_df.columns.values.astype(str)
            column_names = column_names.tolist()
            
            # bge-small-en
            # encoded_input_col = tokenizer(column_names, padding=True, truncation=True, return_tensors='pt')
            # encoded_input_col = {k: v.cuda() for k, v in encoded_input_col.items()}
            # output = self.model.module.st(**encoded_input_col)
            # col_name_emb = F.normalize(output[0][:, 0], p=2, dim=1)
            
            # all-minilm
            encoded_input_col = tokenizer(column_names, padding=True, truncation=True, return_tensors='pt')
            encoded_input_col = {k: v.cuda() for k, v in encoded_input_col.items()}
            encoded_output_col = self.model.module.st(**encoded_input_col)
            encoded_output_col = mean_pooling(encoded_output_col, encoded_input_col['attention_mask'])
            col_name_emb = F.normalize(encoded_output_col, p=2, dim=1)
            
            table_emb = table_emb.cpu()
            col_name_emb = col_name_emb.cpu()

            if add_col_emb:
                origin_table_emb = table_emb.copy()
                column_names = table_df.columns.to_list()
                col_name_emb = self.model.encode(column_names) # NOTE: may have bug related to torch/numpy
                for j, row in enumerate(table):
                    table_emb[j] += col_name_emb
        else:
            table_emb = np.load(table_file)
            
            # Row/column truncation + shuffle
            row_truncation = np.random.permutation(range(table_emb.shape[0]))[:self.max_row]
            table_emb = table_emb[row_truncation, :, :]

            column_truncation = np.random.permutation(range(table_emb.shape[1]))[:self.max_col]
            table_emb = table_emb[:, column_truncation, :]

        

        

        if add_col_emb:
            col_name_emb = col_name_emb[column_truncation, :]
            origin_table_emb = origin_table_emb[row_truncation, :, :]
            origin_table_emb = origin_table_emb[:, column_truncation, :]
            return table_emb, origin_table_emb, col_name_emb
        elif self.pred_type == 'contrastive':
            return table_emb, col_name_emb
        else:
            return table_emb 

if __name__ == "__main__":
    generate_special_tokens()
    cls_token, sep_token, shuffle_tokens = get_special_tokens()
    shuffle_tokens_mat = shuffle_tokens[:3].unsqueeze(1).repeat(1, 3, 1)
    print(shuffle_tokens_mat)
