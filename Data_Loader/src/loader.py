import warnings
warnings.filterwarnings('ignore')

import os
import shutil
import yaml
from pathlib import Path
import pandas as pd
import datasets as ds
from tqdm import tqdm
from Data_Loader import chras_translation
from utils.utils import remove_ptn1, clean_data


CONFIGURATIONS = Path(r'config\configurations.yaml')

class DataLoader:

    '''Loads the data from huggingface hub and validates the records according to patterns'''

    def __init__(self):
        with open(CONFIGURATIONS, 'r') as cfgs_file:
            self.cfgs = yaml.safe_load(cfgs_file)    
            
        # extract the required values from configurations
        self.data_repo = self.cfgs['Data_Source']['src_repo']
        self.root_dir = os.path.join(self.cfgs['Artifacts']['root_dir'], self.cfgs['Data_Loader']['ign_root_dir'])
        self.splits = self.cfgs['Data_Source']['split']
        self.feature = self.cfgs['Data_Source']['feature']
        self.cols = self.cfgs['Data_Source']['cols_key']
        
        # delete the variable after extracting the values
        del self.cfgs
        
        # creates the data folder where data will store. if exist, delete the folder or it's content for loading new data
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir)


    def load_data(self):
        hf_data = ds.load_dataset(self.data_repo)
        
        # now save the all loaded splits, defualt .py data
        for split in tqdm(self.splits, desc="Load & Clean data: ", colour="green"):
            data = pd.DataFrame(hf_data[split][self.feature])
            
            data = clean_data(data)
            
            data[self.cols[1]] = data[self.cols[1]].apply(remove_ptn1)
            data.dropna(ignore_index=True, inplace=True)
            data.to_csv(os.path.join(self.root_dir, f'{split}.csv'), index=False)
            
        for key, value in tqdm(chras_translation.items(), desc="Load Defualt Data: ", colour="green"):
            data = pd.DataFrame(value)
            data.to_csv(os.path.join(self.root_dir, f'{key}.csv'), index=False)            

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()            
            
            