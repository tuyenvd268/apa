import glob
import os
import argparse
from pathlib import Path

import convert_chain_to_pytorch


def makedirs_for_file(acoustic_model_path):
    path = Path(acoustic_model_path)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)

def download_librispeech_models(librispeech_models_path):
    if not os.path.exists(librispeech_models_path):
        os.makedirs("librispeech_models/")
        os.system("wget https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz")
        os.system("wget https://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz")
        os.system("wget https://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz")
        os.system("tar -xf 0013_librispeech_v1_chain.tar.gz -C " + librispeech_models_path)
        os.system("tar -xf 0013_librispeech_v1_lm.tar.gz -C " + librispeech_models_path)
        os.system("tar -xf 0013_librispeech_v1_extractor.tar.gz -C " + librispeech_models_path)
        os.system("rm -f 0013_librispeech_v1_chain.tar.gz")
        os.system("rm -f 0013_librispeech_v1_lm.tar.gz")
        os.system("rm -f 0013_librispeech_v1_extractor.tar.gz")

def prepare_pytorch_models( , libri_chain_mdl_path, libri_chain_txt_path, acoustic_model_path):
    #Convert librispeech acoustic model .mdl to .txt
    if not os.path.exists(libri_chain_txt_path):
        os.system("nnet3-copy --binary=false " + libri_chain_mdl_path + " " + libri_chain_txt_path)

    #Create directory for pytorch models
    if not os.path.exists(pytorch_models_path):
        os.makedirs(pytorch_models_path)

    #Convert final.txt to pytorch acoustic model used in alginments stage
    if not os.path.exists(acoustic_model_path):
        makedirs_for_file(acoustic_model_path)
        config_dict = {
            "libri-chain-txt-path": libri_chain_txt_path,
            "acoustic-model-path":  acoustic_model_path
        }
        convert_chain_to_pytorch.main(config_dict)


def main(config_dict):
    librispeech_models_path = config_dict['librispeech-models-path']
    libri_chain_mdl_path = config_dict['libri-chain-mdl-path']
    libri_chain_txt_path = config_dict['libri-chain-txt-path']
    
    acoustic_model_path = config_dict['acoustic-model-path']
    pytorch_models_path = config_dict['pytorch-models-path']
    
    #Download librispeech models and extract them into librispeech-models-path
    download_librispeech_models(librispeech_models_path)

    #Prepare pytorch models
    prepare_pytorch_models(pytorch_models_path, libri_chain_mdl_path, libri_chain_txt_path, acoustic_model_path)


if __name__ == "__main__":
    librispeech_models_path
    
    
    main(config_dict)