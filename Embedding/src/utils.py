import os
import pathlib as plb
import shutil
import sys
import re



def make_fldrs(paths:list[plb.Path]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        
        
def is_valid_row(text):
    # Unicode range for Devanagari script (Hindi): \u0900-\u097F
    allowed_symbols = r'€£¥§³ØĐǆΩ∞°±µ²³‰₹¢ƒ•¶™©®√π'
    english_hindi_pattern = rf'^[\w\s\u0900-\u097F.,!?@#%&*()_+=|:;<>/\\\'\"\-\[\]({allowed_symbols})]*$'

    # Regex pattern to match HTML tags
    html_pattern = r'<[^>]+>'

    # Combined pattern: matches either HTML tags or non-allowed characters
    combined_pattern = rf'({html_pattern})|([^a-zA-Z0-9\u0900-\u097F\s.,!?@#%&*()_+=|:;<>/\\\'\"\-\[\]({allowed_symbols})])'

    # Check if the text contains any non-allowed characters or HTML tags
    return not bool(re.search(combined_pattern, text))