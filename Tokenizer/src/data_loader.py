import os
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm
from Tokenizer.src import hi_chrs


CONFIG_FILE = Path(r"config\configurations.yaml")

class DataProcessor:
    def __init__(self, config_filepth = CONFIG_FILE):
        self.config = config_filepth
        
        with open(self.config, 'r') as file:
            self.cfgs = yaml.safe_load(file)
            file.close()
                    
        self.data_bkt = self.cfgs['Tokenizer']['src_data_dir']
        self.temp_bkt = os.path.join(self.cfgs['Tokenizer']['root_dir'], self.cfgs['Tokenizer']['temp_data_dir'])
        os.makedirs(self.temp_bkt, exist_ok=True)
        
        self.sub_bkts = self.cfgs['Tokenizer']['sub_folders']
        
        for bkt in self.sub_bkts:
            os.makedirs(os.path.join(self.temp_bkt, bkt), exist_ok=True)
                    
        
    def process_data(self):
        dfs = []
        files = os.listdir(self.data_bkt)
        
        for file in files:
            path = os.path.join(self.data_bkt, file)
            df = pd.read_csv(path)
            dfs.append(df)
    
        for bkt in self.sub_bkts:
            
            print(f"Loading {bkt} data...")
            
            for i, df in tqdm(enumerate(dfs)):
                lines = [str(line) + " \n" for line in list(df[bkt])]
                with open(os.path.join(self.temp_bkt, bkt, f"{i}.txt"), 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
            if bkt == 'hi':
                with open(os.path.join(self.temp_bkt, bkt, f"chrs.txt"), 'w', encoding='utf-8') as f:
                    f.writelines([hi_chr + " \n" for hi_chr in hi_chrs])
                
                
if __name__ == "__main__":
    pass
    # loader = DataLoader()
    # loader.load_data()