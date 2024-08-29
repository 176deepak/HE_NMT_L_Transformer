import os
from pathlib import Path
import yaml

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace

from src.data_loader import DataLoader
from src.bpe import BPE_Tokenizer


CONFIG_FILE = Path(r"config\configurations.yaml")
with open(CONFIG_FILE, 'r') as file:
    cfgs = yaml.safe_load(file)
    file.close()
    

# step 1: Data Loading from data source
data_loader = DataLoader()
data_loader.load_data()
# step 1: end


# step 2: Tokenizer training start
TOKENIZER_CKPTS = 'tokenizer_ckpts'
os.makedirs(TOKENIZER_CKPTS, exist_ok=True)
lang_keys = cfgs['Tokenizer']['lang_keys']
data_bkt = cfgs['Tokenizer']['data_igt_bkt']


for lang_key in lang_keys:
    # step 2.1: BPE Tokenizer training start

    os.makedirs(os.path.join(TOKENIZER_CKPTS, lang_key), exist_ok=True)

    if lang_key == 'en':
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
        tokenizer = bpe_tokenizer.train(os.path.join(data_bkt, lang_key))
        tokenizer.save(os.path.join(TOKENIZER_CKPTS, lang_key, 'tokenizer.json'))
        
    if lang_key == 'hi':
        bpe_tokenizer = BPE_Tokenizer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
            unk_token="[UNK]", 
            pre_tokenizer=Whitespace(), 
            vocab_size=2_08_000, 
            min_frequency=1, 
            continuing_subword_prefix="##",
            max_token_length=10
        )    
        tokenizer = bpe_tokenizer.train(os.path.join(data_bkt, lang_key))
        tokenizer.save(os.path.join(TOKENIZER_CKPTS, lang_key, 'tokenizer.json'))

