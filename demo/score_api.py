import librosa
import torch
import yaml
import json
import re

from src.score_model import GOPT
from src.wavlm_model import (
    WavLM, WavLMConfig
)

def load_config(path):
    config_fh = open(path, "r")
    config_dict = yaml.safe_load(config_fh)
    return config_dict

def init_models(configs, gopt_ckpt_path=None, wavlm_ckpt_path=None):
    gopt_model = GOPT(
        embed_dim=configs['embed-dim'], 
        num_heads=configs['num-heads'], 
        depth=configs['depth'], 
        input_dim=configs['input-dim'], 
        max_length=configs['max-length'], 
        num_phone=configs['num-phone'])

    if gopt_ckpt_path is not None:
        gopt_state_dict = torch.load(gopt_ckpt_path, map_location="cpu")
        gopt_model.load_state_dict(gopt_state_dict)

    assert wavlm_ckpt_path is not None
    if wavlm_ckpt_path is not None:
        wavlm_state_dict = torch.load(wavlm_ckpt_path, map_location="cpu")

        wavlm_config = WavLMConfig(wavlm_state_dict['cfg'])
        wavlm_model = WavLM(wavlm_config)

        wavlm_model.load_state_dict(wavlm_state_dict['model'])

    return gopt_model, wavlm_model, wavlm_config

def extract_feature(alignment, features):
    index = 0
    phonemes = []
    indices = -1 * torch.ones(alignment[-1][1] + alignment[-1][2])
    for phoneme, start_frame, duration in alignment:
        end_frame = start_frame + duration
        indices[start_frame:end_frame] = index
        phonemes.append(phoneme)
        index += 1

    indices[indices==-1] = indices.max() + 1

    indices = torch.nn.functional.one_hot(indices.long(), num_classes=int(indices.max().item())+1)
    indices = indices / indices.sum(0, keepdim=True)
    
    if features.shape[0] != indices.shape[0]:
        features = features[0:indices.shape[0]]
    features = torch.matmul(indices.transpose(0, 1), features)

    return features, phonemes

class Score_Model():
    def __init__(self, configs, gopt_ckpt_path, \
        wavlm_ckpt_path, phone_dict_path):
        self.configs = configs

        self.gopt_model, self.wavlm_model, self.wavlm_config = init_models(
            configs=configs,
            gopt_ckpt_path=gopt_ckpt_path,
            wavlm_ckpt_path=wavlm_ckpt_path)

        self.phone_dict = json.load(open(phone_dict_path, "r"))

    def run(self, waveform, alignments, gop_features):
        phones = [re.sub("\d", "", phone[0]) for phone in alignments]
        phone_ids = [self.phone_dict[phone] for phone in phones]
        durations = [round(phone[2] * 0.02, 4) for phone in alignments]

        phone_ids = torch.tensor(phone_ids).unsqueeze(0)
        waveform = torch.tensor(waveform).unsqueeze(0)
        gop_features = torch.tensor(gop_features).unsqueeze(0)
        durations = torch.tensor(durations).unsqueeze(0)

        wavlm_features = self.run_extract_feature(waveform, alignments)

        utterance_scores, phone_score, word_scores = self.run_scoring(
            wavlm_features=wavlm_features.unsqueeze(0), 
            gop_features=gop_features, 
            durations=durations.unsqueeze(-1),
            phone_ids=phone_ids)

        return utterance_scores, phone_score, word_scores

    def run_scoring(self, wavlm_features, gop_features, phone_ids, durations):
        features = torch.concat(
            [
                gop_features, 
                durations, 
                wavlm_features
            ], dim=-1)

        utterance_scores, phone_score, word_scores = self.gopt_model(
            x=features, phn=phone_ids)

        return utterance_scores, phone_score, word_scores

    @torch.no_grad()
    def run_extract_feature(self, waveform, alignment):
        features = self.wavlm_model.extract_features(waveform)[0]

        index = torch.arange(features.shape[1]).unsqueeze(-1)
        expanded_index = index.expand((-1, 2)).flatten()
        
        features = features[0][expanded_index]
        features, phonemes = extract_feature(alignment, features)

        return features[0:len(phonemes)]

if __name__ == "__main__":
    configs = load_config("configs/model.yaml")

    wavlm_ckpt_path = "/data/codes/prep_ps_pykaldi/pretrained/wavlm-base+.pt"
    gopt_ckpt_path = "/data/codes/prep_ps_pykaldi/exp/ckpts/model.pt"
    phone_dict_path = "/data/codes/prep_ps_pykaldi/resources/phone_dict.json"
    
    score_model = Score_Model(
        configs=configs, 
        phone_dict_path=phone_dict_path,
        gopt_ckpt_path=gopt_ckpt_path, 
        wavlm_ckpt_path=wavlm_ckpt_path)

    inputs = json.load(open("/data/codes/prep_ps_pykaldi/inputs.json"))
    waveform, sample_rate = librosa.load("/data/codes/prep_ps_pykaldi/demo/wav/test.wav", sr=16000)

    gop_features = inputs["gop_features"]
    alignments = inputs["alignments"]
    phonemes = inputs["phonemes"]

    utterance_scores, phone_scores, word_scores = \
        score_model.run(waveform, alignments, gop_features)

    print(utterance_scores.shape)
    print(phone_scores.shape)
    print(word_scores.shape)
