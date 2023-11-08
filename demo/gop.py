from scipy.special import softmax
from kaldiio import ReadHelper
from tqdm import tqdm
from glob import glob
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import pickle
import yaml
import os

from utils.gop import (
    generate_df_phones_pure,
    gop_robust_with_matrix,
    prepare_dataframes,
    get_alignments,
    pad_loglikes
)

class GOP(object):
    def __init__(self, configs):
        self.phones_path = configs["phones_path"]
        self.final_mdl_path = configs["final_mdl_path"]
        self.phones_pure_path = configs["phones_pure_path"]
        self.phones_to_pure_int_path = configs["phones_to_pure_int_path"]

        self.transition_dir = 'exp/'

        self.df_phones_pure = self.prepare_df_phones_pure()

    def prepare_df_phones_pure(self):
        phones_pure_path = generate_df_phones_pure(
            phones_path=self.phones_path, 
            phones_to_pure_int_path=self.phones_to_pure_int_path, 
            phones_pure_path=self.phones_pure_path, 
            final_mdl_path=self.final_mdl_path, 
            transition_dir=self.transition_dir
        )

        df_phones_pure = pd.read_pickle(phones_pure_path)
        df_phones_pure = df_phones_pure.reset_index()

        return df_phones_pure

    def prepare_df_alignments(self, alignments_path):
        alignments_dict = get_alignments(alignments_path)

        return alignments_dict
    
    def run(self, align_path="exp/align.out", prob_path="exp/prob.ark"):
        df_alignments = self.prepare_df_alignments(align_path)
        df_alignments = pd.DataFrame.from_dict(df_alignments)

        gop_dict = self.compute_gop(prob_path=prob_path, df_alignments=df_alignments)
        return gop_dict

    def compute_gop(self, prob_path, df_alignments):
        gop_dict = {}
        
        df_scores = df_alignments.transpose()
        with ReadHelper('ark:' + prob_path) as reader:
            for i, (key, loglikes) in enumerate(tqdm(reader)):
                loglikes = softmax(np.array(loglikes), axis=1)
                df_scores_batch = df_scores.iloc[i:i+1]

                assert key == df_scores_batch.index[0]
                loglikes_batch  = [loglikes, ]
                padded_loglikes = pad_loglikes(loglikes_batch)
                df_scores_batch['p'] = padded_loglikes

                gop_dict = gop_robust_with_matrix(
                    df_scores_batch, 
                    self.df_phones_pure, 
                    number_senones=6024, 
                    batch_size=len(df_scores_batch), 
                    output_gop_dict=gop_dict)
        
        return gop_dict

if __name__ == '__main__':
    prob_path = 'exp/prob.ark'
    gop_path = 'exp/gop.pkl'

    configs = {
        'phones_path': 'kaldi/exp/chain_cleaned/tdnn_1d_sp/phones.txt',
        'final_mdl_path': 'kaldi/exp/chain_cleaned/tdnn_1d_sp/final.mdl',
        'phones_pure_path': 'kaldi/data/phones/phones-list.txt',
        'phones_to_pure_int_path': 'kaldi/data/phones/phone-to-pure-phone.int',
    }

    gop = GOP(configs=configs)

    gop_dict  = gop.run(
        align_path="exp/align.out", 
        prob_path="exp/prob.ark"
    )
    
    print(gop_dict)