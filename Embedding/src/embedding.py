import os
import shutil
from pathlib import Path
import re
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from gensim.models import Word2Vec
from tqdm import tqdm


CONFIG_FILE = Path(r"config\config.yaml")


class Word2Vec:
    def __init__(self, tokenizer_cfg):
        