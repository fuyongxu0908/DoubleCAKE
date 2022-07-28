from torch.utils.data import Dataset, DataLoader


class TrainSet(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size):

        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        return positive_sample

    def __len__(self):
        return len(self.triples)