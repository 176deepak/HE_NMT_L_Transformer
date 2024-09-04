from tokenizers import Tokenizer

# tokenization layer
class TokenizationLayer:
    def __init__(self, cfgs):
        self.tokenizer = Tokenizer.from_file(cfgs)
        self.padding=False
        self.pad_length = None
        self.truncation = False
        self.trunc_length = None
        self.add_start = False
        self.add_end = False
        
    def set_config(self, pad_length=None, trunc_length=None, add_start=False, add_end=False):
        if add_start:
            self.add_start = True
        if add_end:
            self.add_end = True
        if pad_length:
            self.padding = True
            self.tokenizer.enable_padding(
                direction='right',
                pad_id=3,
                pad_type_id=0,
                pad_token='[PAD]',
                length=pad_length
            )
        if trunc_length:
            self.truncation = True
            if add_end:
                trunc_length -= 1
            self.tokenizer.enable_truncation(max_length=trunc_length, direction='right')
            
    def encode_batch(self, batch):
        if self.add_start:
            batch = [f"[SOS] {seq}" for seq in batch]
        
        encoded_batch = self.tokenizer.encode_batch(batch)
        encoded_batch = [encoded_seq.tokens for encoded_seq in encoded_batch]
        
        if self.add_end: 
            for tokens in encoded_batch:
                if '[PAD]' in tokens:
                    idx = tokens.index('[PAD]')
                    tokens[idx] = '[EOS]'
                else:
                    tokens.append('[EOS]')
        
        return encoded_batch