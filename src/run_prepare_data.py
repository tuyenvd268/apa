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

from acoustic_models import FTDNNAcoustic

def load_data(data_dir):
    metadata_path = f'{data_dir}/metadata.csv'
    wav_dir = f'{data_dir}/wav'
    
    data = pd.read_csv(metadata_path, names=["path", "text"], sep="|")
    data["path"] = data.path.apply(lambda x: os.path.join(wav_dir, f'{x}.wav'))
    data["id"] = data.path.apply(lambda x: os.path.basename(x).split(".wav")[0])
    
    return data

def save_kaldi_data_format(data, data_dir):
    data.sort_values("id")
    
    wavscp_path = f'{data_dir}/wav.scp'
    text_path = f'{data_dir}/text'
    spk2utt_path = f'{data_dir}/spk2utt'
    utt2spk_path = f'{data_dir}/utt2spk'
    wavscp_file = open(wavscp_path, "w", encoding="utf-8")
    text_file = open(text_path, "w", encoding="utf-8")
    spk2utt_file = open(spk2utt_path, "w", encoding="utf-8")
    utt2spk_file = open(utt2spk_path, "w", encoding="utf-8")
    
    for index in data.index:
        wavscp = f'{data["id"][index]}\t{data["path"][index]}\n'
        text = f'{data["id"][index]}\t{data["text"][index]}\n'
        spk2utt = f'{data["id"][index]}\t{data["id"][index]}\n'
        utt2spk = f'{data["id"][index]}\t{data["id"][index]}\n'
        
        wavscp_file.write(wavscp)
        text_file.write(text)
        spk2utt_file.write(spk2utt)
        utt2spk_file.write(utt2spk)
        
    wavscp_file.close()
    text_file.close()
    spk2utt_file.close()
    utt2spk_file.close()

def load_config(path):
    config_fh = open(path, "r")
    config_dict = yaml.safe_load(config_fh)
    return config_dict

def extract_features_using_kaldi(conf_path, wav_scp_path, spk2utt_path, mfcc_path, ivectors_path, feats_scp_path):
    os.system(
        'compute-mfcc-feats --config='+conf_path+'/mfcc_hires.conf \
            scp,p:' + wav_scp_path+' ark:- | copy-feats \
            --compress=true ark:- ark,scp:' + mfcc_path + ',' + feats_scp_path)
        
    os.system(
        'ivector-extract-online2 --config='+ conf_path +'/ivector_extractor.conf ark:'+ spk2utt_path + '\
            scp:' + feats_scp_path + ' ark:' + ivectors_path)

def parallel_extract_feature(conf_path, n_split, split_dir):
    for index in tqdm(range(n_split)):
        sub_data_dir = f'{split_dir}/{index}'

        wav_scp_path = f'{sub_data_dir}/wav.scp'
        spk2utt_path = f'{sub_data_dir}/spk2utt'
        mfcc_path = f'{sub_data_dir}/mfcc.{index}.ark'
        ivectors_path = f'{sub_data_dir}/ivectors.{index}.ark'
        feats_scp_path = f'{sub_data_dir}/feats.{index}.scp'

        extract_features_using_kaldi(conf_path, wav_scp_path, spk2utt_path, mfcc_path, ivectors_path, feats_scp_path)

def split_data(data, n_split, out_dir):
    splits = os.listdir(out_dir)
    
    n_sample_per_split = int(data.shape[0] / n_split)
    for index in range(n_split):
        sub_data_dir = f'{out_dir}/{index}'
        if not os.path.exists(sub_data_dir):
            os.mkdir(sub_data_dir)

        sub_data = data[index*n_sample_per_split: (index+1)*n_sample_per_split]
        save_kaldi_data_format(sub_data, sub_data_dir)

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
     
def init_dir(split_dir):
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)
    else:
        sub_folders = os.listdir(split_dir)
        for folder in sub_folders:
            shutil.rmtree(os.path.join(split_dir, folder))

def load_feature(logid, conf_path, mfcc_path, ivectors_path):
    ivector_period = load_ivector_period_from_conf(conf_path)

    mfccs_rspec = ("ark:" + mfcc_path)
    ivectors_rspec = ("ark:" + ivectors_path)
    with RandomAccessMatrixReader(mfccs_rspec) as mfccs_reader, \
        RandomAccessMatrixReader(ivectors_rspec) as ivectors_reader:

        mfccs = mfccs_reader[logid]
        ivectors = ivectors_reader[logid]
        ivectors = np.repeat(ivectors, ivector_period, axis=0) 
        ivectors = ivectors[:mfccs.shape[0],:]
        x = np.concatenate((mfccs,ivectors), axis=1)

        feats = torch.from_numpy(x)

    return feats

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

def parallel_align(config_dict, conf_path, n_split, split_dir, prob_dir, align_dir):
    for index in tqdm(range(n_split), desc="Align"):
        sub_data_dir = f'{split_dir}/{index}'

        wav_scp_path = f'{sub_data_dir}/wav.scp'
        text_path = f'{sub_data_dir}/text'
        mfcc_path = f'{sub_data_dir}/mfcc.{index}.ark'
        ivectors_path = f'{sub_data_dir}/ivectors.{index}.ark'

        prob_path = f'{prob_dir}/output.{index}.ark'
        align_path = f'{align_dir}/align.{index}.out'
        align_v1_path = f'{align_dir}/output.{index}.ali'

        align(config_dict, conf_path, prob_path, align_path, align_v1_path, wav_scp_path, text_path, ivectors_path, mfcc_path)

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

def merge_align(align_dir):
    merged_align = []
    for align_path in glob(f'{align_dir}/output*'):
        align_df = pd.read_csv(align_path, names=["id", "align"], sep="\t")

        merged_align.append(align_df)

    merged_align = pd.concat(merged_align)
    merged_align.reset_index(inplace=True)

    merged_align_path = f'{align_dir}/merged_align.out'
    with open(merged_align_path, "w", encoding="utf-8") as f:
        for index in merged_align.index:
            id = merged_align["id"][index]
            align = merged_align["align"][index]

            line = f'{id}\t{align}\n'
            f.write(line)


if __name__ == '__main__':
    config_dict = load_config("configs/data_prep.yaml")

    n_split = 40
    conf_path = "kaldi/conf"
    data_root_dir = "prep_data/"
    data_dir = "data/"
    split_dir = f'{data_dir}/split'
    exp_dir = 'exp'
    prob_dir = f'{exp_dir}/probs'
    align_dir = f'{exp_dir}/aligns'

    init_dir(split_dir)

    data = load_data(data_root_dir)

    save_kaldi_data_format(data, data_dir)
    split_data(data, n_split, out_dir=split_dir)
    parallel_extract_feature(conf_path, n_split, split_dir)
    parallel_align(config_dict, conf_path, n_split, split_dir, prob_dir, align_dir)
    merge_align(align_dir)
