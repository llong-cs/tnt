import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from dotenv import load_dotenv

load_dotenv()
MODELS_PATH = os.environ["MODELS_PATH"]
DATASETS_PATH = os.environ["DATASETS_PATH"]

# st_name = "paraphrase-MiniLM-L6-v2"
st_name = "all-MiniLM-L6-v2"

if __name__ == '__main__':
    # Load sentence transformer
    model = SentenceTransformer(f'{MODELS_PATH}/{st_name}')
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # spider dataset
    embedding_path = f'{DATASETS_PATH}/embedded_tables_spider_{st_name}'
    embedding_files = os.listdir(embedding_path)

    col_count = {}
    
    for embedding_file in tqdm(embedding_files):
        embedding_file_path = os.path.join(embedding_path, embedding_file)
        table_emb = np.load(embedding_file_path)
        col_num = str(table_emb.shape[1])
        target_dir = os.path.join(embedding_path, col_num)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if not col_num in col_count.keys():
            col_count[col_num] = 1
        with open(os.path.join(target_dir, f'table-{col_count[col_num]}.npy'), 'wb') as f:
            np.save(f, table_emb)
            col_count[col_num] += 1
