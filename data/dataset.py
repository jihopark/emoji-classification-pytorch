from collections import Counter
import pickle
from torch.utils.data import Dataset
from torch import LongTensor
import numpy as np

class EmojiDataset(Dataset):
    def __init__(self, split, max_len=None):
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
        self.data = list(zip(samples, map(lambda x: int(x), labels)))
        print("Data loaded: %s" % len(self.data))
        print("\nData Distribution")
        for key in self.label_dist.keys():
            print("%s: %s (%.2f)" % (self.cluster_emoji[int(key)], self.label_dist[key],
                                    self.label_dist[key]/len(labels)))
        self.max_len = max_len if max_len else max([len(s) for s in samples])
        print("max_len %s" % self.max_len)
        print("samples exceeding max_len %.4f" % ([len(s) > self.max_len for s in samples].count(True)
                                                  / len(samples)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        txt = LongTensor(np.zeros(self.max_len, dtype=np.int64))

        for i, x in enumerate(self.data[idx][0]):
            if i == self.max_len:
                break
            txt[i] = int(x)

        return txt, LongTensor([self.data[idx][1]])
