from .model import *
import torch
from torch.nn import functional as F
import numpy as np
from transformers import AutoModel
from dotenv import load_dotenv
import os

load_dotenv()
DATASETS_PATH = os.environ["DATASETS_PATH"]
MODELS_PATH = os.environ["MODELS_PATH"]
SENTENCE_TRANSFORMER = os.environ["SENTENCE_TRANSFORMER"]

class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class TableEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_cols,
        depth,
        heads,
        dim_head,
        attn_dropout=0.0,
        ff_dropout=0.0,
        attentiontype="col",
        final_mlp_style="common",
        pred_type='generation',
        pooling='cls',
        col_name=False,
        numeric_mlp=False,
        gradient_checkpoint=False
    ):
        super().__init__()

        self.num_cols = num_cols
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style
        self.pred_type = pred_type
        self.cont_dim = 256
        self.pooling = pooling
        self.numeric_mlp = numeric_mlp
        self.gradient_checkpoint = gradient_checkpoint
        
        # initialize sentence transformer
        self.st_name = SENTENCE_TRANSFORMER
        model_dir = f"{MODELS_PATH}/{self.st_name}"
        if self.st_name == 'all-MiniLM-L6-v2' or self.st_name == 'bge-small-en-v1.5':
            self.st = AutoModel.from_pretrained(model_dir)
            self.dim = self.st.config.hidden_size
        elif self.st_name == 'puff-base-v1':
            vector_dim = 768
            self.dim = vector_dim
            self.st = AutoModel.from_pretrained(model_dir)
            self.vector_linear = torch.nn.Linear(in_features=self.st.config.hidden_size, out_features=vector_dim)
            vector_linear_dict = {
                k.replace("linear.", ""): v for k, v in
                torch.load(os.path.join(model_dir, f"2_Dense_{vector_dim}/pytorch_model.bin")).items()
            }
            self.vector_linear.load_state_dict(vector_linear_dict)
        else:
            raise ValueError("Invalid sentence transformer model")        
        
        for param in self.st.parameters():
            param.requires_grad = False
        self.st.pooler = None
        if self.numeric_mlp:
            self.num_mlp = simple_MLP([1, self.dim, self.dim], gradient_checkpoint=self.gradient_checkpoint)
        
        if self.pooling == 'cls':   
            self.cls = nn.Parameter(torch.randn(self.dim))

        # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                dim=self.dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                gradient_checkpoint=self.gradient_checkpoint
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                dim=self.dim,
                nfeats=num_cols,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
                gradient_checkpoint=self.gradient_checkpoint
            )
            
        # projection head
        # needed for contrastive learning
        if self.pred_type == 'contrastive':
            self.col_specific_projection_head = simple_MLP([self.dim, self.dim, self.cont_dim], gradient_checkpoint=self.gradient_checkpoint)
            # self.table_specific_projection_head = simple_MLP([self.dim, self.dim, self.cont_dim // 4])
            if col_name:
                self.col_name_projection_head = nn.Sequential(
                    Transformer(
                        dim=self.dim,
                        depth=1,
                        heads=heads,
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        gradient_checkpoint=self.gradient_checkpoint
                    ),
                    simple_MLP([self.dim, self.dim, self.cont_dim], gradient_checkpoint=self.gradient_checkpoint)
                )
                # self.col_name_projection_head = simple_MLP([self.dim, self.dim, self.cont_dim])
        
        # cell embedding
        # if sentence_transformers is not None:
        #     self.st = sentence_transformers
        #     for param in self.st.parameters():
        #         param.requires_grad = False
        #     self.st.pooler = None
        #     if self.numeric_mlp:
        #         self.num_mlp = simple_MLP([1, self.dim, self.dim])
        
        self.qformer = None
        
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # input_size = [bs, num_rows, num_cols, seq_len]
    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        bs, num_rows, num_cols, seq_len = input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], input_ids.shape[3]
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(-1, seq_len)
        
        if self.st_name == 'all-MiniLM-L6-v2':        
            last_hidden_state = self.st(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            embeddings = self.mean_pooling(last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        elif self.st_name == 'puff-base-v1':
            # puff version
            last_hidden_state = self.st(input_ids=input_ids, attention_mask=attention_mask)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            # mean pooling
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            embeddings = F.normalize(self.vector_linear(vectors), p=2, dim=-1)
        elif self.st_name == 'bge-small-en-v1.5':
            last_hidden_state = self.st(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # cls pooling
            embeddings = last_hidden_state[0][:,0]
            embeddings = F.normalize(embeddings, p=2, dim=-1)
                
        embeddings = embeddings.reshape(bs, num_rows, num_cols, -1)

        return embeddings
                
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        table_mask,
        inference=False
    ):
        if self.pred_type == 'contrastive':
            tab_emb = self.get_embeddings(input_ids, attention_mask, token_type_ids)
                        
            if self.pooling == 'cls':
                # roll the table on dim 1 (row dim)
                tab_emb = torch.roll(tab_emb, 1, 1)
                # insert [cls] token at the first row
                tab_emb[:,0,:,:] = self.cls
                        
            cell_emb = self.transformer(tab_emb, mask=table_mask)
            # [batch_size, num_cols, 384]
            
            if inference:
                for param in self.col_specific_projection_head.parameters():
                     param.requires_grad = True
                col_emb = self.attn_pooling(cell_emb, table_mask)
                return col_emb           
            elif self.pooling == 'cls':
                # the first row is cls -> cls pooling
                col_emb = cell_emb[:,0,:,:]
            else:
                # mean pooling
                col_emb = get_flatten_table_emb(cell_emb, table_mask)
            
            col_spe_cont_emb = F.normalize(self.col_specific_projection_head(col_emb), p=2, dim=-1)
            return col_spe_cont_emb
            
            # table_spe_cont_emb = F.normalize(self.table_specific_projection_head(col_emb), p=2, dim=-1)
            
            # return col_spe_cont_emb, table_spe_cont_emb
        else:
            x = self.transformer(input_ids, mask=table_mask)
            x = self.col_specific_projection_head(x) # [batch_szie, num_rows, num_cols, 384]
            output = get_flatten_table_emb(x, table_mask) # [batch_size, num_cols, 384]

            return output
        
    def unfreeze_st(self):
        for param in self.st.encoder.parameters():
            param.requires_grad = True
        if self.st_name == 'puff-base-v1':
            for param in self.vector_linear.parameters():
                param.requires_grad = True
                
    def attn_pooling(self, cell_emb, table_mask):
        if self.qformer is None:
            self.qformer = Qformer(dim=self.dim, dim_head=64, inner_dim=2048, query_num=5).to(cell_emb.device)
        output = self.qformer(cell_emb, mask=table_mask)
        return output
            