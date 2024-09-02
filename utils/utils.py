import re


validation_pattern = re.compile(r'(<[^>]+>)|([^a-zA-Z0-9\\u0900-\\u097F\\s.,!?@#%&*()_+=|:;<>/\\\\\\\'\\"\\-\\[\\](€£¥§³ØĐǆΩ∞°±µ²³‰₹¢ƒ•¶™©®√π)])')

def validate_text(text):
    return not bool(re.search(validation_pattern, text))


def remove_ptn1(text):
    text = re.sub(r'\(_ [a-zA-Z]\)', '', text)
    text = re.sub(r'\([a-zA-Z] _\)', '', text)
    return text