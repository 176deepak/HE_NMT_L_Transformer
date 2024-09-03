import re


validation_pattern = re.compile(r'(<[^>]+>)|([^a-zA-Z0-9\\u0900-\\u097F\\s.,!?@#%&*()_+=|:;<>/\\\\\\\'\\"\\-\\[\\](€£¥§³ØĐǆΩ∞°±µ²³‰₹¢ƒ•¶™©®√π)])')

def validate_text(text):
    return not bool(re.search(validation_pattern, text))


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