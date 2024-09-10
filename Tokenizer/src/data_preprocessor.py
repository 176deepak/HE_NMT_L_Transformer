import os
import shutil
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

CONFIG_FILE = Path(r"config\configurations.yaml")

class DataProcessor:
    def __init__(self, config_filepth = CONFIG_FILE):
        self.config = config_filepth
        
        with open(self.config, 'r') as file:
            self.cfgs = yaml.safe_load(file)
            file.close()
                    
        self.data_bkt = os.path.join(self.cfgs['Artifacts']['root_dir'], self.cfgs['Tokenizer']['src_data_dir'])
        self.temp_bkt = os.path.join(self.cfgs['Tokenizer']['root_dir'], self.cfgs['Tokenizer']['temp_data_dir'])
        
        if os.path.exists(self.temp_bkt):
            shutil.rmtree(self.temp_bkt)
        os.makedirs(self.temp_bkt)
        
        self.chunk_size = self.cfgs['Tokenizer']['chunk_line']
                    
        
    def process_data(self):
        dfs = [pd.read_csv(os.path.join(self.data_bkt, file)) for file in os.listdir(self.data_bkt)]

        for i, df in tqdm(enumerate(dfs), total=len(dfs), desc=f"CSV data => .txt data: ", ncols=100, colour='green'):
            cols = df.columns.tolist()
            for col in cols:
                lines = [str(line) + " \n" for line in df[col]]
                n_batch = (len(lines) + self.chunk_size - 1) // self.chunk_size
                for j in range(n_batch):
                    s = j * self.chunk_size
                    e = s + self.chunk_size
                    with open(os.path.join(self.temp_bkt, f"{col}_{i}{j}.txt"), 'w', encoding='utf-8') as f:
                        f.writelines(lines[s:e])                                
                
if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data()