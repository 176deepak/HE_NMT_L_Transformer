import os
from pathlib import Path
import datasets as ds
import yaml
from tqdm import tqdm
from utils import make_fldrs


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

        for split in tqdm(splits, desc="Tokenizer Data Loading..."):
            records = data[split]['translation']
            
            for key in self.lang_keys:
                lang_records = [rec[key] for rec in records]
                
                with open(os.path.join(self.data_bkt, key, f"{split}_{key}_corpus.txt"), 'w', encoding='utf8') as file:
                    for line in lang_records:
                        file.write(line.strip("\n") + "\n")
                    file.close()                
                
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()
        

        
        