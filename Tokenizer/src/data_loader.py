import os
from pathlib import Path
import datasets as ds
import yaml
import pandas as pd
from tqdm import tqdm
from src.utils import make_fldrs, is_valid_row
from src import hi_chrs


CONFIG_FILE = Path(r"config\configurations.yaml")

class DataLoader:
    def __init__(self, config_filepth = CONFIG_FILE):
        self.config = config_filepth
        
        with open(self.config, 'r') as file:
            self.cfgs = yaml.safe_load(file)
            file.close()
            
        self.data_repo = self.cfgs['Tokenizer']['hf_data_src']
        
        self.data_bkt = self.cfgs['Tokenizer']['data_igt_bkt']
        os.makedirs(self.data_bkt, exist_ok=True)
        
        self.lang_keys = self.cfgs['Tokenizer']['lang_keys']
        
        paths = []
        for key in self.lang_keys:
            paths.append(Path(f"{self.data_bkt}/{key}"))
        
        make_fldrs(paths=paths)
        
        
    def load_data(self):
        splits = ds.get_dataset_split_names(self.data_repo)
        data = ds.load_dataset(self.data_repo)
        
        # train, validation & test splits
        train_df = pd.DataFrame(data['train']['translation'])
        validation_df = pd.DataFrame(data['validation']['translation'])
        test_df = pd.DataFrame(data['test']['translation'])
        
        train_df = train_df[train_df['en'].apply(is_valid_row) & train_df['hi'].apply(is_valid_row)]
        validation_df = validation_df[validation_df['en'].apply(is_valid_row) & validation_df['hi'].apply(is_valid_row)]
        test_df = test_df[test_df['en'].apply(is_valid_row) & test_df['hi'].apply(is_valid_row)]
        
        dfs = [train_df, validation_df, test_df]

        for col in self.lang_keys:
            
            print(f"Loading {col} data...")
            
            for i, df in tqdm(enumerate(dfs)):
                lines = [str(line) + " \n" for line in list(df[col])]
                with open(os.path.join(self.data_bkt, col, f"{i}.txt"), 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                f.close()
                
            if col == 'hi':
                with open(os.path.join(self.data_bkt, col, f"chrs.txt"), 'w', encoding='utf-8') as f:
                    f.writelines([hi_chr + " \n" for hi_chr in hi_chrs])

                
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()
        

        
        