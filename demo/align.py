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
import base64
import json
import yaml
import sys
import os

from src.acoustic_model import FTDNNAcoustic

def load_config(path):
    config_fh = open(path, "r")
    configs = yaml.safe_load(config_fh)
    return configs

def load_ivector_period_from_conf(conf_path):
    conf_fh = open(conf_path + '/ivector_extractor.conf', 'r')
    ivector_period_line = conf_fh.readlines()[1]
    ivector_period = int(ivector_period_line.split('=')[1])
    return ivector_period

def prepare_data_in_kaldi_format(data_dir, text, wav_path, ID=8888):
    with open(f'{data_dir}/wav.scp', "w", encoding="utf-8") as wavscp_file:
        wavscp_file.write(f'{ID}\t{wav_path}\n')

    with open(f'{data_dir}/text', "w", encoding="utf-8") as text_file:
        text_file.write(f'{ID}\t{text}\n')

    with open(f'{data_dir}/spk2utt', "w", encoding="utf-8") as spk2utt_file:
        spk2utt_file.write(f'{ID}\t{ID}\n')

    with open(f'{data_dir}/utt2spk', "w", encoding="utf-8") as utt2spk_file:
        utt2spk_file.write(f'{ID}\t{ID}\n')
    
    print(f'###saved data to {data_dir} ')

def log_alignments(aligner, phones, alignment, logid, align_output_fh):
    phone_alignment = aligner.to_phone_alignment(alignment, phones)
    
    transition_lists = []
    for phone, start_time, duration in phone_alignment:
        transitions_for_phone = alignment[start_time : (start_time + duration)]
        transition_lists.append(transitions_for_phone)
    align_output_fh.write(logid + ' phones '      + str(phone_alignment)  + '\n')
    align_output_fh.write(logid + ' transitions ')

    for transition_list in transition_lists:
        align_output_fh.write(str(transition_list) + ' ')
    align_output_fh.write('\n')

    return logid, phone_alignment

def extract_features_using_kaldi(conf_path, wav_scp_path, spk2utt_path, mfcc_path, ivectors_path, feats_scp_path):
    os.system(
        'compute-mfcc-feats --config='+conf_path+'/mfcc_hires.conf \
            scp,p:' + wav_scp_path+' ark:- | copy-feats \
            --compress=true ark:- ark,scp:' + mfcc_path + ',' + feats_scp_path)
        
    os.system(
        'ivector-extract-online2 --config='+ conf_path +'/ivector_extractor.conf ark:'+ spk2utt_path + '\
            scp:' + feats_scp_path + ' ark:' + ivectors_path)

def initialize(transition_model_path, tree_path, lang_graph_path, \
    words_path, disam_path, phones_path, word_boundary_path, acoustic_model_path, num_senones):

    aligner = MappedAligner.from_files(
        transition_model_path, tree_path, 
        lang_graph_path, words_path,
        disam_path, beam=40.0, acoustic_scale=1.0)
    
    phones  = SymbolTable.read_text(phones_path)
    word_boundary_info = WordBoundaryInfo.from_file(
        WordBoundaryInfoNewOpts(),
        word_boundary_path)

    acoustic_model = FTDNNAcoustic(num_senones=num_senones)
    acoustic_model.load_state_dict(torch.load(acoustic_model_path))
    acoustic_model.eval()

    return aligner, phones, word_boundary_info, acoustic_model

class Aligner(object):
    def __init__(self, configs):
        self.data_dir = configs["data-dir"]
        self.exp_dir = configs["exp-dir"]
        self.conf_path = configs["conf-path"]

        self.wav_scp_path = f'{self.data_dir}/wav.scp'
        self.text_path = f'{self.data_dir}/text'
        self.spk2utt_path = f'{self.data_dir}/spk2utt'
        self.mfcc_path = f'{self.data_dir}/mfcc.ark'
        self.ivectors_path = f'{self.data_dir}/ivectors.ark'
        self.feats_scp_path = f'{self.data_dir}/feats.scp'

        self.prob_path = f'{self.exp_dir}/prob.ark'
        self.align_path = f'{self.exp_dir}/align.out'
        self.align_feature_path = f'{self.exp_dir}/output.ali'

        self.acoustic_model_path = configs['acoustic-model-path']
        self.transition_model_path = configs['transition-model-path']
        self.tree_path = configs['tree-path']
        self.disam_path = configs['disambig-path']
        self.word_boundary_path = configs['word-boundary-path']
        self.lang_graph_path = configs['lang-graph-path']
        self.words_path = configs['words-path']
        self.phones_path = configs['kaldi-phones-path']
        self.num_senones = configs['num_senones']

        self.aligner, self.phones, self.word_boundary_info, self.acoustic_model = \
            initialize(
                transition_model_path=self.transition_model_path, 
                tree_path=self.tree_path, 
                lang_graph_path=self.lang_graph_path, 
                words_path=self.words_path, 
                disam_path=self.disam_path, 
                phones_path=self.phones_path, 
                word_boundary_path=self.word_boundary_path, 
                acoustic_model_path=self.acoustic_model_path,
                num_senones=self.num_senones
            )
        self.ivector_period = load_ivector_period_from_conf(self.conf_path)

    def run(self, wav_path, text):
        prepare_data_in_kaldi_format(
            data_dir=self.data_dir,
            wav_path=wav_path,
            text=text
        )

        extract_features_using_kaldi(
            conf_path=self.conf_path, 
            wav_scp_path=self.wav_scp_path, 
            spk2utt_path=self.spk2utt_path, 
            mfcc_path=self.mfcc_path, 
            ivectors_path=self.ivectors_path, 
            feats_scp_path=self.feats_scp_path
        )

        alignments = self.run_align(
            prob_path=self.prob_path, 
            align_path=self.align_path)

        return alignments

    def run_align(self, prob_path, align_path):
        prob_wspec= f"ark:{prob_path}"
        align_file = open(align_path,"w+")

        text_df = pd.read_csv(
            self.text_path, names=["id", "text"], 
            sep="\t", index_col=0).to_dict()["text"]
        
        mfccs_rspec = ("ark:" + self.mfcc_path)
        ivectors_rspec = ("ark:" + self.ivectors_path)
        mfccs_reader = RandomAccessMatrixReader(mfccs_rspec)
        ivectors_reader = RandomAccessMatrixReader(ivectors_rspec)
        prob_writer = DoubleMatrixWriter(prob_wspec)

        wav_list_paths = open(self.wav_scp_path, "r").readlines()
        assert len(wav_list_paths) == 1
        
        for line in tqdm(wav_list_paths, desc="Align"):
            logid, _ = line.split("\t")
            text = text_df[int(logid)].upper()

            mfccs = mfccs_reader[logid]
            ivectors = ivectors_reader[logid]
            ivectors = np.repeat(ivectors, self.ivector_period, axis=0) 
            ivectors = ivectors[:mfccs.shape[0],:]
            x = np.concatenate((mfccs,ivectors), axis=1)

            feats = torch.from_numpy(x).unsqueeze(0)

            with torch.no_grad():
                loglikes = self.acoustic_model(feats)

            loglikes = Matrix(loglikes.detach().numpy()[0])
            prob_writer[logid] = loglikes
            output = self.aligner.align(loglikes, text)
            logid, phone_alignment = log_alignments(
                self.aligner, self.phones, output["alignment"], logid, align_file)

            phone_alignment = self.aligner.to_phone_alignment(output["alignment"], self.phones)

        prob_writer.close()
        align_file.close()

        return phone_alignment

if __name__ == "__main__":
    configs = load_config("config.yaml")

    wav_path = "wav/1403449.wav"
    text = "CHILLY"

    aligner = Aligner(configs)
    aligner.run(wav_path, text)