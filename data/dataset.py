from collections import Counter
import pickle
from torch.utils.data import Dataset
from torch import LongTensor
import numpy as np
import logging

class EmojiDataset(Dataset):
    def __init__(self, split, max_len=None):
        self.split = split
        self.vocab = pickle.load(open("./data/new_twitter.vocab","rb"))
        logging.info("----------%s--------------", split)
        logging.info("Vocab loaded: %s words" % len(self.vocab["word2index"]))
        if split == "train":
            self.pretrained_embeddings = np.load("./data/twitter.glove.npy")
            logging.info("loaded pretrained embedding %s" % str(self.pretrained_embeddings.shape))

        self.cluster_emoji = [line.rstrip().replace("\t","") for line in open("./data/cluster_emoji.txt", "r")]

        samples = [line.rstrip().split("\t") for line in open("./data/idx_emoji_%s.tsv" % split, "r")]
        labels = [line.rstrip() for line in open("./data/label_%s.tsv" % split, "r")]

        self.label_dist = Counter(labels)
        self.data = list(zip(samples, map(lambda x: int(x), labels)))
        logging.info("Data loaded: %s" % len(self.data))
        for key in self.label_dist.keys():
            logging.info("%s: %s (%.2f)" % (self.cluster_emoji[int(key)], self.label_dist[key],
                                    self.label_dist[key]/len(labels)))
        self.max_len = max_len if max_len else max([len(s) for s in samples])
        logging.debug("max_len %s" % self.max_len)
        logging.debug("samples exceeding max_len %.4f" % ([len(s) > self.max_len for s in samples].count(True)
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
