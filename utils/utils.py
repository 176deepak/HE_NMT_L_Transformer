import re
import numpy as np
import pandas as pd


validation_pattern = r'(<[^>]+>)|([^a-zA-Z0-9\u0900-\u097F\s[!-\/:-@[-`{-~€£¥§³ØĐǆΩ∞°±µ²³‰₹¢ƒ•¶™©®√π])'

# def validate_text(text):
#     return not bool(re.search(validation_pattern, text))


def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    
    # remove empty strings or empty cells
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)    
    df.dropna(inplace=True, ignore_index=True)
    
    # validate text
    mask = df[cols].apply(lambda col: ~col.astype(str).str.contains(validation_pattern, regex=True))
    
    # Filter rows where all columns satisfy the condition
    df = df[mask.all(axis='columns')]
    
    df = df.reset_index(drop=True)
    
    return df
        
    
def remove_ptn1(text):
    text = re.sub(r'\(_ [a-zA-Z]\)', '', text)
    text = re.sub(r'\([a-zA-Z] _\)', '', text)
    return text


def batch_generator(seqs, batch_size=32):
    batch = []
    for seq in seqs:
        batch.append(seq)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
        
        
def tokens_for_embeddings(tokenizer, seqs):
    encoded_seqs = tokenizer.encode_batch(seqs)
    return [encoded_seq.tokens for encoded_seq in encoded_seqs]