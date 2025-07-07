import numpy as np
import pandas as pd
import torch
import os
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
DATASETS_PATH = os.environ["DATASETS_PATH"]
MODELS_PATH = os.environ["MODELS_PATH"]

st_name = "all-MiniLM-L6-v2"


def list_csv_file(data_path):
    csv_files = []
    files_list = os.listdir(data_path)
    for files in files_list:
        file_list = os.listdir(f"{data_path}/{files}")
        for file in file_list:
            if file.endswith(".csv"):
                csv_files.append(f"{data_path}/{files}/{file}")
    return csv_files


class TableDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, train=True, pred_type="rank_index"):
        self.data_path = data_path
        self.table_list = list_csv_file(data_path)
        if train:
            self.table_list = self.table_list[: -len(self.table_list) // 10]
        else:
            self.table_list = self.table_list[-len(self.table_list) // 10 :]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.pred_type = pred_type

    def __len__(self):
        return len(self.table_list)

    def __getitem__(self, idx):
        table_df = pd.read_csv(
            self.table_list[idx],
            encoding="utf-8",
            low_memory=False,
        )

        if len(table_df) > 100:
            table_df = table_df.sample(n=100)

        table = []
        table_token_num = []
        col_num = len(table_df.columns)
        max_token_num = [0 for _ in range(col_num)]
        for i, row in table_df.iterrows():
            row_data = []
            row_token_num = []
            for j in len(table_df.columns):
                value = table_df.at[i, j]
                if isinstance(value, np.number):
                    row_data.append(value)
                    row_token_num.append(1)
                    max_token_num[j] = 1
                else:
                    tokens = self.tokenizer.encode(str(value), return_tensors="np")
                    row_data.append(tokens.squeeze())
                    row_token_num.append(len(tokens))
                    if max_token_num[j] < len(tokens):
                        max_token_num[j] = len(tokens)
            table.append(row_data)
            table_token_num.append(row_token_num)
                
        # pad to max num
        mask = []
        for i in range(len(table)):
            row_mask = []
            for j in range(col_num):
                value = table[i][j]
                if isinstance(value, np.number):
                    row_mask.append(1)
                else:
                    # TODO
                    pass
        return table, mask

