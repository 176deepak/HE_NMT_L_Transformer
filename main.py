import os
from pathlib import Path
import yaml
import shutil
import argparse
from Data_Loader.src.loader import DataLoader
from Tokenizer.src.data_preprocessor import DataProcessor
from Tokenizer.src.bpe import BPE_Tokenizer
from Embedding.src.embedding import WordEmbedding
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tqdm import tqdm


# get the terminal width(cols) for better visualitation of execution
trml_size = shutil.get_terminal_size()
trml_cols = trml_size.columns


# add arguments in script execution command, which will help in for deciding whether the tokenizer and embeddings training
parser = argparse.ArgumentParser(description="Run specific code blocks based on boolean flags.")
parser.add_argument("--LoadData", action="store_true", help="Helps to execute the Data Loader")
parser.add_argument("--TrainTokenizer", action="store_true", help="Helps to execute the Tokenizer step & Embedding Step")
parser.add_argument("--TrainEmbedding", action="store_true", help="Helps to execute the Embedding Step")
args = parser.parse_args()


# read the configuration files
CONFIG_FILE = Path(r"config\configurations.yaml")
with open(CONFIG_FILE, 'r') as file:
    cfgs = yaml.safe_load(file)
    
# create the artifacts folder, where all the artifacts will store
os.makedirs(cfgs['Artifacts']['root_dir'], exist_ok=True)
    

# Step 1: Load the Data from src + Data Cleaning
if args.LoadData:
    print("\nData Loading: ")
    print("+"*trml_cols)
    loader = DataLoader()
    loader.load_data()
    print("+"*trml_cols, end="\n\n")


# Step 2: Train Tokenizer on loaded data, if you want
if args.TrainTokenizer:
    print("\nTrain Tokenizer: ")
    print("+"*trml_cols)
    data_processor = DataProcessor()
    data_processor.process_data()
    print("\n")

    TOKENIZER_CKPTS = os.path.join(cfgs['Artifacts']['root_dir'], cfgs['Tokenizer']['ckpts_bkt'])
    os.makedirs(TOKENIZER_CKPTS, exist_ok=True)
    # sub_dirs = cfgs['Tokenizer']['sub_folders']
    data_bkt = os.path.join(cfgs['Tokenizer']['root_dir'], cfgs['Tokenizer']['temp_data_dir'])

    bpe_tokenizer = BPE_Tokenizer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"], 
        unk_token="[UNK]", 
        # norm_seq=normalizers.Sequence([NFD(), StripAccents()]), 
        pre_tokenizer=Whitespace(), 
        vocab_size=4_16_000, 
        min_frequency=2, 
        continuing_subword_prefix="##",
        max_token_length=10
    )    
    tokenizer = bpe_tokenizer.train(data_bkt)
    tokenizer.save(os.path.join(TOKENIZER_CKPTS, 'tokenizer.json'))
    
    shutil.rmtree(data_bkt)
    print("+"*trml_cols, end="\n\n")

# Step 3: Train Word2Vec embedding model on custom tokens data, if you want
if args.TrainEmbedding:
    print("\nTrain Word Embedding: ")
    print("+"*trml_cols)
    word2vec = WordEmbedding()
    tokenizer = Tokenizer.from_file(os.path.join(cfgs['Artifacts']['root_dir'], cfgs['Tokenizer']['ckpts_bkt'], 'tokenizer.json'))
    tokenizer.enable_padding(
        direction='right',
        pad_id=3,
        pad_type_id=0,
        pad_token='[PAD]',
        length=None)
    word2vec.train_embedding(tokenizer=tokenizer, flag='hi')
    print('\n')
    print("+"*trml_cols)
    
# Step 4: Train Transformer model on custom dataset with using trained/pre-trained tokenizer and word embeddings.