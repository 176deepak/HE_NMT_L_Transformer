import os
import pathlib as plb
import shutil
import sys



def make_fldrs(paths:list[plb.Path]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        