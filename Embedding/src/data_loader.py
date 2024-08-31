import os
from pathlib import Path
import datasets as ds
import yaml
import pandas as pd
from tqdm import tqdm
from utils import make_fldrs, is_valid_row


CONFIG_FILE = Path(r"config\config.yaml")

class DataLoader:
    def __init__(self, config_filepth = CONFIG_FILE):
        self.config = config_filepth
        
        with open(self.config, 'r') as file:
            self.cfgs = yaml.safe_load(file)
            file.close()
            
        self.data_repo = self.cfgs['word2vec']['hf_data_src']
        self.data_bkt = self.cfgs['word2vec']['data_igt_bkt']
        os.makedirs(self.data_bkt, exist_ok=True)
                
        
    def load_data(self):
        data = ds.load_dataset(self.data_repo)
        
        # train, validation & test splits
        train_df = pd.DataFrame(data['train']['translation'])
        validation_df = pd.DataFrame(data['validation']['translation'])
        test_df = pd.DataFrame(data['test']['translation'])
        
        train_df = train_df[train_df['en'].apply(is_valid_row) & train_df['hi'].apply(is_valid_row)]
        validation_df = validation_df[validation_df['en'].apply(is_valid_row) & validation_df['hi'].apply(is_valid_row)]
        test_df = test_df[test_df['en'].apply(is_valid_row) & test_df['hi'].apply(is_valid_row)]
        
        df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
        
        df.to_csv(os.path.join(self.data_bkt, 'data.csv'), index=False)        
        

                
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()
        

        
        