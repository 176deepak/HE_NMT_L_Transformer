import os
from pathlib import Path
import yaml
import shutil
import argparse
from Data_Loader.src.loader import DataLoader
from Tokenizer.src.data_loader import DataProcessor
from Tokenizer.src.bpe import BPE_Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace


parser = argparse.ArgumentParser(description="Run specific code blocks based on boolean flags.")
# Add boolean arguments
parser.add_argument("--TrainTokenizer", action="store_true", help="Helps to execute the Tokenizer step")
parser.add_argument("--TrainEmbedding", action="store_true", help="Helps to execute the Embedding Step")
args = parser.parse_args()


CONFIG_FILE = Path(r"config\configurations.yaml")
with open(CONFIG_FILE, 'r') as file:
    cfgs = yaml.safe_load(file)
    

# Step 1: Load the Data from src + Data Cleaning
# loader = DataLoader()
# loader.load_data()


# Step 2: Train Tokenizer on loaded data, if you want
if args.TrainTokenizer:
    data_processor = DataProcessor()
    data_processor.process_data()

    TOKENIZER_CKPTS = os.path.join(cfgs['Tokenizer']['root_dir'], cfgs['Tokenizer']['ckpts_bkt'])
    os.makedirs(TOKENIZER_CKPTS, exist_ok=True)
    sub_dirs = cfgs['Tokenizer']['sub_folders']
    data_bkt = os.path.join(cfgs['Tokenizer']['root_dir'], cfgs['Tokenizer']['temp_data_dir'])

    for dir in sub_dirs:

        os.makedirs(os.path.join(TOKENIZER_CKPTS, dir), exist_ok=True)

        if dir == 'en':
            bpe_tokenizer = BPE_Tokenizer(
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
                unk_token="[UNK]", 
                norm_seq=normalizers.Sequence([NFD(), StripAccents()]), 
                pre_tokenizer=Whitespace(), 
                vocab_size=2_08_000, 
                min_frequency=2, 
                continuing_subword_prefix="##",
                max_token_length=10
            )    
            tokenizer = bpe_tokenizer.train(os.path.join(data_bkt, dir))
            tokenizer.save(os.path.join(TOKENIZER_CKPTS, dir, 'tokenizer.json'))
        
        if dir == 'hi':
            bpe_tokenizer = BPE_Tokenizer(
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
                unk_token="[UNK]", 
                pre_tokenizer=Whitespace(), 
                vocab_size=2_08_000, 
                min_frequency=1, 
                continuing_subword_prefix="##",
                max_token_length=10
            )    
            tokenizer = bpe_tokenizer.train(os.path.join(data_bkt, dir))
            tokenizer.save(os.path.join(TOKENIZER_CKPTS, dir, 'tokenizer.json'))
            
    shutil.rmtree(data_bkt)

# Step 3: Train Word2Vec embedding model on custom tokens data, if you want
if args.TrainEmbedding:
    pass

# Step 4: Train Transformer model on custom dataset with using trained/pre-trained tokenizer and word embeddings.