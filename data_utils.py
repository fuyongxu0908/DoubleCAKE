import json

from torch.utils.data import Dataset, DataLoader
import pickle
import os


class FB17K_237(Dataset):
    def __init__(self, args, type):
        self.args = args
        with open(os.path.join(args.save_path, type + '_p_n_triples.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample

    def __len__(self):
        return len(self.dataset)