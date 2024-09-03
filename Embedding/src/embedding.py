import os
import yaml
import re
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from utils.utils import batch_generator, tokens_for_embeddings


CONFIGURATIONS = Path(r'config\configurations.yaml')

class WordEmbedding:
    def __init__(self, config_path=CONFIGURATIONS):
        self.config_path = config_path
        
        with open(self.config_path, 'r') as f:
            self.cfgs = yaml.safe_load(f)
        
        self.src_data = os.path.join(self.cfgs['Artifacts']['root_dir'], self.cfgs['Embedding']['src_data_dir'])
        self.ckpts_dir = os.path.join(self.cfgs['Artifacts']['root_dir'], self.cfgs['Embedding']['ckpts_bkt'])
        self.sub_dirs = self.cfgs['Embedding']['sub_folders']
        
        os.makedirs(self.ckpts_dir, exist_ok=True)

        for dir in self.sub_dirs:
            os.makedirs(os.path.join(self.ckpts_dir, dir), exist_ok=True)
            
        self.files = os.listdir(self.src_data)
        
        self.dfs = []
        for file in self.files:
            file_path = os.path.join(self.src_data, file)
            self.dfs.append(pd.read_csv(file_path))
        
        self.df = pd.concat(self.dfs, ignore_index=True)
        self.df.dropna(ignore_index=True, inplace=True)
            
        del self.dfs
            
    def train_embedding(self, tokenizer, flag):
        col = flag
        sentences = list(self.df[col])
        seqs_of_tokens = []
        
        batch_size = 32
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        
        for batched_seqs in tqdm(batch_generator(sentences), desc=f"{col} Tokens Generation:", colour='green', ncols=100, total=total_batches):
            batched_tokens = tokens_for_embeddings(tokenizer, batched_seqs)
            seqs_of_tokens.extend(batched_tokens)
        model = Word2Vec(sentences=seqs_of_tokens, vector_size=512, seed=42, workers=3, sg=0, )
        model.save(os.path.join(self.ckpts_dir, col, 'model.model'))            

