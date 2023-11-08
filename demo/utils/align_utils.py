from kaldi.util.table import RandomAccessMatrixReader
from kaldi.util.table import DoubleMatrixWriter
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.matrix import Matrix
from kaldi.lat.align import (
    WordBoundaryInfoNewOpts, 
    WordBoundaryInfo)

from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import argparse
import shutil
import torch
import json
import yaml
import sys
import os

def load_config(path):
    config_fh = open(path, "r")
    config_dict = yaml.safe_load(config_fh)
    return config_dict

def load_ivector_period_from_conf(conf_path):
    conf_fh = open(conf_path + '/ivector_extractor.conf', 'r')
    ivector_period_line = conf_fh.readlines()[1]
    ivector_period = int(ivector_period_line.split('=')[1])
    return ivector_period


def prepare_data_in_kaldi_format(data_dir, text, wav_path):
    ID = 8888
    with open(f'{data_dir}/wav.scp', "w", encoding="utf-8") as wavscp_file:
        wavscp_file.write(f'{ID}\t{wav_path}\n')

    with open(f'{data_dir}/text', "w", encoding="utf-8") as text_file:
        text_file.write(f'{ID}\t{text}\n')

    with open(f'{data_dir}/spk2utt', "w", encoding="utf-8") as spk2utt_file:
        spk2utt_file.write(f'{ID}\t{ID}\n')

    with open(f'{data_dir}/utt2spk', "w", encoding="utf-8") as utt2spk_file:
        utt2spk_file.write(f'{ID}\t{ID}\n')
    
    print(f'###saved data to {data_dir} ')