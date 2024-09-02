import os
import re
import yaml
from pathlib import Path
import pandas as pd
import datasets as ds
from tqdm import tqdm
from ...utils.utils import validate_text, remove_ptn1


CONFIGURATIONS = Path(r'config\configurations.yaml')

# class DataLoader:
#     def __init__(self):
#         with open(CONFIGURATIONS, 'r') as cfgs_file:
#             self.cfgs = yaml.safe_load(cfgs_file)
            
#         self.root_dir = self.cfgs['Data_Loader']['ign_root_dir']
        
#         os.makedirs(self.root_dir, exist_ok=True)

#     def load_data(self):
#         hf_data = ds.load_dataset(self.cfgs['Data_Source']['src_repo'])
#         feature = self.cfgs['Data_Source']['feature']
        
#         for split in tqdm(self.cfgs['Data_Source']['split'], desc="Loading & Processing data: ", colour="green"):
#             data = pd.DataFrame(hf_data[split][feature])
            
#             cols = self.cfgs['Data_Source']['cols_key']
#             data = data.loc[data[cols].apply(lambda row: all(validate_text(row[col]) for col in cols), axis=1)]
            
#             data[cols[1]] = data[cols[1]].apply(remove_ptn1)
#             data.to_csv(os.path.join(self.root_dir, f'{split}.csv'), index=False)
                        

if __name__ == "__main__":
    # loader = DataLoader()
    # loader.load_data()
    
    print("Success")
            
            
            