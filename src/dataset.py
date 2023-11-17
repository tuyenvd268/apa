from torch.utils.data import Dataset, DataLoader
import torch

class PrepDataset(Dataset):
    def __init__(self, phone_ids, word_ids, phone_scores, word_scores, sentence_scores, durations, gops, wavlm_features, arpas_categories):
        self.phone_ids = phone_ids
        self.word_ids = word_ids

        self.phone_scores = phone_scores
        self.word_scores = word_scores
        self.sentence_scores = sentence_scores

        self.gops = gops
        self.durations = durations
        self.wavlm_features = wavlm_features
        self.arpas_categories =arpas_categories
        
    def __len__(self):
        return self.phone_ids.shape[0]
    
    def parse_data(self, phone_ids, word_ids, phone_scores, word_scores, sentence_scores, durations, gops, wavlm_features, arpas_categories):
        phone_ids = torch.tensor(phone_ids)
        word_ids = torch.tensor(word_ids)

        phone_scores = torch.tensor(phone_scores).float().clone()
        word_scores = torch.tensor(word_scores).float().clone()
        sentence_scores = torch.tensor(sentence_scores).float().clone()
        arpas_categories = torch.tensor(arpas_categories).float().clone()

        phone_scores[phone_scores != -1] /= 50
        word_scores[word_scores != -1] /= 50
        sentence_scores /= 50

        durations = torch.tensor(durations)
        gops = torch.tensor(gops)
        wavlm_features = torch.tensor(wavlm_features)

        features = torch.concat([gops, durations.unsqueeze(-1), wavlm_features], dim=-1)        
        return {
            "features": features,
            "phone_ids": phone_ids,
            "word_ids": word_ids,
            "phone_scores":phone_scores,
            "word_scores":word_scores,
            "sentence_scores":sentence_scores,
            "arpas_categories":arpas_categories
        }
        
    def __getitem__(self, index):
        phone_ids = self.phone_ids[index]
        word_ids = self.word_ids[index]

        phone_scores = self.phone_scores[index]
        word_scores = self.word_scores[index]
        sentence_scores = self.sentence_scores[index]

        gops = self.gops[index]
        durations = self.durations[index]
        wavlm_features = self.wavlm_features[index]
        arpas_categories = self.arpas_categories[index]

        return self.parse_data(
            phone_ids=phone_ids,
            word_ids=word_ids,
            phone_scores=phone_scores,
            word_scores=word_scores,
            sentence_scores=sentence_scores,
            gops=gops,
            durations=durations,
            wavlm_features=wavlm_features,
            arpas_categories=arpas_categories
        )

if __name__ == "__main__":
    data_dir = "/data/codes/prep_ps_pykaldi/exp/sm/test"

    phone_ids, word_ids, phone_scores, word_scores, sentence_scores, durations, gops, wavlm_features = load_data(data_dir)
    dataset = PrepDataset(phone_ids, word_ids, phone_scores, word_scores, sentence_scores, durations, gops, wavlm_features)
    dataloader = DataLoader(dataset, batch_size=8)

    for batch in dataloader:
        features = batch["features"]
        phone_ids = batch["phone_ids"]
        word_ids = batch["word_ids"]

        phone_scores = batch["phone_scores"]
        word_scores = batch["word_scores"]
        sentence_scores = batch["sentence_scores"]
        
        print(features.shape)
        print(phone_ids.shape)
        print(word_ids.shape)
        print(phone_scores.shape)
        break

    dataset = None
    dataloader = None