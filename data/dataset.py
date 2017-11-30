from collections import Counter
import pickle
from torch.utils.data import Dataset
import numpy as np

class EmojiDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.vocab = pickle.load(open("./data/new_twitter.vocab","rb"))
        print("Vocab loaded: %s words" % len(self.vocab["word2index"]))
        if split == "train":
            self.pretrained_embeddings = np.load("./data/twitter.glove.npy")
            print("loaded pretrained embedding %s" % str(self.pretrained_embeddings.shape))

        self.cluster_emoji = [line.rstrip().replace("\t","") for line in open("./data/cluster_emoji.txt", "r")]
        for i, cluster in enumerate(self.cluster_emoji):
            print("%s: %s" % (i, cluster))

        samples = [line.rstrip().split("\t") for line in open("./data/idx_emoji_%s.tsv" % split, "r")]
        labels = [line.rstrip() for line in open("./data/label_%s.tsv" % split, "r")]

        self.label_dist = Counter(labels)
        self.data = list(zip(samples, labels))
        print("Data loaded: %s" % len(self.data))
        print("\nData Distribution")
        for key in self.label_dist.keys():
            print("%s: %s (%.2f)" % (self.cluster_emoji[int(key)], self.label_dist[key],
                                    self.label_dist[key]/len(labels)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
