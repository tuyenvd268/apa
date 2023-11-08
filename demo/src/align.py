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

from src.acoustic_models import FTDNNAcoustic

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

def load_ivector_period_from_conf(conf_path):
    conf_fh = open(conf_path + '/ivector_extractor.conf', 'r')
    ivector_period_line = conf_fh.readlines()[1]
    ivector_period = int(ivector_period_line.split('=')[1])
    return ivector_period

def align(config_dict, conf_path, prob_path, align_path, align_v1_path, wav_list_path, text_path, ivectors_path, mfcc_path):
    aligner, phones, wb_info, acoustic_model = initialize(config_dict)
    ivector_period = load_ivector_period_from_conf(conf_path)

    prob_wspec= f"ark:| copy-feats --compress=true ark:- ark:{prob_path}"
    align_file = open(align_path,"w+")
    align_v1_file = open(align_v1_path,"w+")

    text_df = pd.read_csv(text_path, names=["id", "text"], sep="\t", index_col=0)
    text_df = text_df.to_dict()["text"]

    mfccs_rspec = ("ark:" + mfcc_path)
    ivectors_rspec = ("ark:" + ivectors_path)
    mfccs_reader = RandomAccessMatrixReader(mfccs_rspec)
    ivectors_reader = RandomAccessMatrixReader(ivectors_rspec)
    prob_writer = DoubleMatrixWriter(prob_wspec)

    for line in tqdm(open(wav_list_path, "r").readlines(), desc="Align"):
        logid, _ = line.split("\t")
        text = text_df[int(logid)].upper()

        mfccs = mfccs_reader[logid]
        ivectors = ivectors_reader[logid]
        ivectors = np.repeat(ivectors, ivector_period, axis=0) 
        ivectors = ivectors[:mfccs.shape[0],:]
        x = np.concatenate((mfccs,ivectors), axis=1)

        feats = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            loglikes = acoustic_model(feats)
        loglikes = Matrix(loglikes.detach().numpy()[0])
        prob_writer[logid] = loglikes
        output = aligner.align(loglikes, text)
        logid, phone_alignment = log_alignments(
            aligner, phones, output["alignment"], logid, align_file)

        phone_alignment = aligner.to_phone_alignment(output["alignment"], phones)

        json_obj = json.dumps(phone_alignment)
        align_v1_file.write(f'{logid}\t{json_obj}\n')

    prob_writer.close()
    align_v1_file.close()
    align_file.close()

def load_config(path):
    config_fh = open(path, "r")
    config_dict = yaml.safe_load(config_fh)
    return config_dict

def initialize(config_dict):
    acoustic_model_path = 'kaldi/torch/acoustic_model.pt'
    transition_model_path = config_dict['transition-model-path']
    tree_path = config_dict['tree-path']
    disam_path = config_dict['disambig-path']
    word_boundary_path = config_dict['word-boundary-path']
    lang_graph_path = config_dict['lang-graph-path']
    words_path = config_dict['words-path']
    phones_path = config_dict['libri-phones-path']
        
    aligner = MappedAligner.from_files(
        transition_model_path, tree_path, 
        lang_graph_path, words_path,
        disam_path, acoustic_scale=1.0)
    
    phones  = SymbolTable.read_text(phones_path)
    wb_info = WordBoundaryInfo.from_file(
        WordBoundaryInfoNewOpts(),
        word_boundary_path)

    acoustic_model = FTDNNAcoustic()
    acoustic_model.load_state_dict(torch.load(acoustic_model_path))
    acoustic_model.eval()

    return aligner, phones, wb_info, acoustic_model


if __name__ == "__main__":
    config_dict = load_config("configs/data_prep.yaml")

    data_dir = "data/"
    wav_path = "wav/1403449.wav"
    text = "CHILLY"

    os.system(". ./path.sh")

    prepare_data_in_kaldi_format(
        data_dir=data_dir,
        wav_path=wav_path,
        text=text
    )

    conf_path = "../kaldi/conf"
    wav_scp_path = "data/wav.scp"
    spk2utt_path = "data/spk2utt"
    mfcc_path = "data/mfcc.ark"
    ivectors_path = "data/ivectors.ark"
    feats_scp_path = "data/feats.scp"

    extract_features_using_kaldi(
        conf_path=conf_path, 
        wav_scp_path=wav_scp_path, 
        spk2utt_path=spk2utt_path, 
        mfcc_path=mfcc_path, 
        ivectors_path=ivectors_path, 
        feats_scp_path=feats_scp_path
    )

    wav_scp_path = 'data/wav.scp'
    text_path = 'data/text'
    mfcc_path = 'data/mfcc.ark'
    ivectors_path = 'data/ivectors.ark'

    prob_path = 'exp/output.ark'
    align_path = 'exp/align.out'
    align_v1_path = 'exp/output.ali'


    align(
        config_dict=config_dict, 
        conf_path=conf_path, 
        prob_path=prob_path, 
        align_path=align_path, 
        align_v1_path=align_v1_path, 
        wav_list_path=wav_scp_path, 
        text_path=text_path, 
        ivectors_path=ivectors_path, 
        mfcc_path=mfcc_path
    )
