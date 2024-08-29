import os
from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class BPE_Tokenizer:
    def __init__(self, 
                special_tokens=[], 
                unk_token="[UNK]", 
                norm_seq=None, 
                pre_tokenizer=Whitespace(), 
                vocab_size=30_000, 
                min_frequency=2, 
                continuing_subword_prefix="##",
                max_token_length=10
        ):
        self.special_tokens = special_tokens
        self.unk_token = unk_token
        self.norm_seq = norm_seq
        self.pre_tokenizer = pre_tokenizer
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.continuing_subword_prefix = continuing_subword_prefix
        self.max_token_length = max_token_length
        
        
    def train(self, data_src):
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        tokenizer.normalizer = self.norm_seq
        tokenizer.pre_tokenizer = self.pre_tokenizer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens = self.special_tokens,
            continuing_subword_prefix = self.continuing_subword_prefix,
            max_token_length = self.max_token_length
        )
        
        files = [os.path.join(data_src, file) for file in os.listdir(data_src)]
        
        print(f"Tokenizer training start...")
        tokenizer.train(files, trainer)
        print(f"Tokenizer training end...")
        
        # tokenizer.save(json_save_dir)
        return tokenizer
        
        

if __name__ == "__main__":
    bpe_tokenizer = BPE_Tokenizer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
        unk_token="[UNK]", 
        norm_seq=normalizers.Sequence([NFD(), StripAccents()]), 
        pre_tokenizer=Whitespace(), 
        vocab_size=30_000, 
        min_frequency=2, 
        continuing_subword_prefix="##",
        max_token_length=10
    )
    
    tokenizer = bpe_tokenizer.train(r'data\en')
    tokenizer.save('tokenizer.json')