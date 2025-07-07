import numpy as np
import pandas as pd
import torch
from torch import nn, einsum
import torch.optim as optim
import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import TableEncoder
import wandb
from utils import *
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--log_activate", type=bool, default=False)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--save_path", type=str, default="checkpoints")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", type=bool, default=True)
parser.add_argument("--drop_last", type=bool, default=False)

# attention param
parser.add_argument('--num_columns', default=100, type=int)
parser.add_argument('--embedding_size', default=384, type=int)
parser.add_argument('--transformer_depth', default=12, type=int)
parser.add_argument('--attention_heads', default=6, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--dim_head', default=64, type=int)

# training param
parser.add_argument("--pred_type", type=str, default="generation")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--load_model", type=bool, default=False)
parser.add_argument("--model_path", type=str)

def list_npy_files(data_path):
    file_list = os.listdir(data_path)
    npy_files = []
    for file in file_list:
        if file.endswith(".npy"):
            npy_files.append(file)
    return npy_files

def list_csv_files(data_path):
    file_list = os.listdir(data_path)
    csv_files = []
    for file in file_list:
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files

def list_csv_files_from_database(data_path):
    file_list = os.listdir(data_path)
    csv_files = []
    for file_name in file_list:
        file_path = os.path.join(data_path, file_name)
        table_list = os.listdir(file_path)
        for table in table_list:
            table_path = os.path.join(file_path, table)
            if table.endswith(".csv"):
                csv_files.append(table_path)
    return csv_files

def get_special_tokens():
    special_tokens = np.load("data/special_tokens.npz")
    return (
        special_tokens["cls_token"],
        special_tokens["sep_token"],
        special_tokens["shuffle_tokens"],
    )

class TableDataset(Dataset):
    def __init__(self, data_path, from_csv=False, model=None):
        self.data_path = data_path
        self.from_csv = from_csv
        if self.from_csv:
            self.table_list = list_csv_files_from_database(self.data_path)
        else:
            self.table_list = list_npy_files(self.data_path)
        self.cls_token, self.sep_token, self.shuffle_tokens = get_special_tokens()
        self.model = model

    def __len__(self):
        return len(self.table_list)

    def __getitem__(self, idx):
        # Read table embeddings
        row_size = 50
        column_size = 50
        
        if self.from_csv:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            
            table_df = pd.read_csv(self.table_list[idx], encoding='utf-8', low_memory=False)
            
            if len(table_df) > row_size:
                table_df = table_df.sample(n=row_size)
            table = table_df.to_numpy()
            table = table.astype(str)
            table_emb = np.zeros((table.shape[0], table.shape[1], embedding_dim))
            for j, row in enumerate(table):
                row_emb = self.model.encode(row)
                table_emb[j] = row_emb
        else:
            table_emb = np.load(os.path.join(self.data_path, self.table_list[idx]))
        
            # Row/column truncation + shuffle
            row_truncation = np.random.permutation(range(table_emb.shape[0]))[:row_size]
            table_emb = table_emb[row_truncation, :, :]
        
        column_truncation = np.random.permutation(range(table_emb.shape[1]))[:column_size]
        table_emb = table_emb[:, column_truncation, :]

        table_emb_padded = np.zeros((row_size, 100, table_emb.shape[2]))
        table_emb_padded[:table_emb.shape[0], :table_emb.shape[1], :] = table_emb
        table_mask = np.zeros((row_size, 100))
        table_mask[:table_emb.shape[0], :table_emb.shape[1]] = 1
        
        return table_emb_padded, table_mask, self.table_list[idx]

def generate_table_embedding(args):
    sentence_model = SentenceTransformer("path_of_paraphrase-MiniLM-L6-v2")
    dataset = TableDataset(data_path=args.data_path, from_csv=True, model=sentence_model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = TableEncoder(
        num_cols=args.num_columns,
        dim=args.embedding_size,
        depth=args.transformer_depth,
        heads=args.attention_heads,
        attn_dropout=args.attention_dropout,
        ff_dropout=args.ff_dropout,
        attentiontype="colrow",
        decode=False,
        pred_type='contrastive',
        dim_head=args.dim_head
    )
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda()
    for table_emb, table_mask, table_path in tqdm(dataloader):
        table_emb = table_emb.cuda()
        table_mask = table_mask.cuda()
        with torch.no_grad():
            table_emb, table_mask = table_emb.float().cuda(), table_mask.float().cuda()
            output = model(anchor_table=table_emb, anchor_table_mask=table_mask, inference=True)
            for i, path in enumerate(table_path):
                mask = table_mask[i,0,:].cpu().numpy()
                record = output[i].cpu().numpy()[np.where(mask == 1)]
                print(record.shape)
                dir_path = os.path.join("~/database_emb", path.split("/")[-2])
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                np.save(os.path.join(dir_path, path.split("/")[-1].replace(".csv", ".npy")), record)
                
def generate_table_embedding_for_contrastive_comparison(args):
    stat = torch.zeros(1, 100)
    sentence_model = SentenceTransformer("path_of_paraphrase-MiniLM-L6-v2")
    dataset = TableDataset(data_path=args.data_path, from_csv=True, model=sentence_model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = TableEncoder(
        num_cols=args.num_columns,
        dim=args.embedding_size,
        depth=args.transformer_depth,
        heads=args.attention_heads,
        attn_dropout=args.attention_dropout,
        ff_dropout=args.ff_dropout,
        attentiontype="colrow",
        decode=False,
        pred_type='contrastive',
        dim_head=args.dim_head
    )
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for table_emb, table_mask, table_path in tqdm(dataloader):
        table_emb = table_emb.cuda()
        table_mask = table_mask.cuda()
        with torch.no_grad():
            table_emb, table_mask = table_emb.float().cuda(), table_mask.float().cuda()
            output = model(anchor_table=table_emb, anchor_table_mask=table_mask, inference=True)
            for i in range(len(output)):
                stat_row = torch.zeros(1, 100)
                mask = table_mask[i,0,:]
                emb_with_context = output[i][torch.where(mask == 1)]
                single_table_emb = table_emb[i] # [50, 100, 384]
                single_table_mask = table_mask[i] # [50, 100]
                col_num = torch.sum(single_table_mask, axis=1)[0].int().item()
                if col_num == 0:
                    continue
                single_column_mask = torch.zeros((50, 100))
                single_column_mask[:,0] = single_table_mask[:,0]
                single_column_mask = single_column_mask.unsqueeze(0)
                single_column_mask = single_column_mask.repeat(col_num, 1, 1)
                seperate_table_emb = torch.zeros((col_num, 50, 100, 384))
                for j in range(col_num):
                    seperate_table_emb[j,:,0,:] = single_table_emb[:,j,:]
                seperate_table_emb, single_column_mask = seperate_table_emb.cuda(), single_column_mask.cuda()
                output_single_column = model(anchor_table=seperate_table_emb, anchor_table_mask=single_column_mask, inference=True) # [n, 100, 384]
                emb_without_context = torch.zeros_like(emb_with_context) # [n, 384]
                for j in range(col_num):
                    emb_without_context[j] = output_single_column[j,0,:]
                emb_with_context = F.normalize(emb_with_context, p=2, dim=-1)
                emb_without_context = F.normalize(emb_without_context, p=2, dim=-1)
                
                sim = cos(emb_with_context, emb_without_context)
                avg_sim = torch.mean(sim)
                stat_row[0,col_num] = avg_sim
                stat = torch.cat((stat, stat_row), dim=0)
    stat_non_zero = torch.count_nonzero(stat, dim=0)
    stat = torch.sum(stat, dim=0)
    avg_stat = stat/stat_non_zero
    avg_stat[torch.isnan(avg_stat)] = 0
    x = range(100)
    plt.plot(x, avg_stat, 's-',color = 'r')
    plt.xlabel('Column Number')
    plt.ylabel('Average Cosine Similarity')
    plt.savefig('figs/cont_sim.jpg')
    
                    
            

if __name__ == '__main__':
    args = parser.parse_args()
    # generate_table_embedding(args)
    generate_table_embedding_for_contrastive_comparison(args)
            
            
       
