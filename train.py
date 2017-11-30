import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import classification_report, f1_score

from model.lstm_clf import LSTMClassifier
from data.dataset import EmojiDataset
import config_helper

CONFIG_KEYS = ["model_name", "hidden_dim", "num_layers", "max_len",
               "config_file", "batch_size", "learning_rate", "max_epochs",
               "log_path", "print_every"]

parser =argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
args = vars(parser.parse_args())

FLAGS = config_helper.load(args["config_file"], CONFIG_KEYS)
print(FLAGS)

# prepare the data
train_dataset = EmojiDataset("train", max_len=FLAGS["max_len"])
valid_dataset = EmojiDataset("valid", max_len=FLAGS["max_len"])

train_loader = DataLoader(train_dataset, batch_size=FLAGS["batch_size"],
                          shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=FLAGS["batch_size"],
                          shuffle=True, num_workers=4)

if FLAGS["model_name"] == "bi-lstm":
    model = LSTMClassifier(hidden_dim=FLAGS["hidden_dim"],
                           vocab_size=len(train_dataset.vocab["word2index"]),
                           target_size=len(train_dataset.label_dist.keys()),
                           batch_size=FLAGS["batch_size"],
                           num_layers=FLAGS["num_layers"],
                           pretrained_embedding=train_dataset.pretrained_embeddings,
                           embedding_dim=train_dataset.pretrained_embeddings.shape[1])
    model.cuda()
    print(model)

optimizer = optim.SGD(model.parameters(), lr=FLAGS["learning_rate"])
loss_function = nn.CrossEntropyLoss()

for epoch in range(FLAGS["max_epochs"]):
    total_loss = []
    total_preds = []
    total_truth = []

    for step, (x, y) in enumerate(train_loader):
        y = torch.squeeze(y)
        x, y = Variable(x.cuda()), y.cuda()

        model.zero_grad()
        model.hidden = model.init_hidden()

        output = model(x.t())
        loss = loss_function(output, Variable(y))
        loss.backward()
        optimizer.step()

        # calc training acc
        _, predicted = torch.max(output.data, 1)
        total_preds += predicted.tolist()
        total_truth += y.tolist()
        total_loss.append(loss.data[0])

        if step and step % FLAGS["print_every"] == 0:
            print("epoch %s-step %s/%s=\
                    loss: %.4f, f1:%.4f" % (epoch, step, len(train_loader),
                                            np.mean(total_loss),
                                            f1_score(total_truth, total_preds, average="weighted")))
            total_loss = []
            total_preds = []
            total_truth = []

    # evaluate at end of every epoch
    valid_loss = []
    valid_preds = []
    valid_truth = []
    for x, y in valid_loader:
        y = torch.squeeze(y)
        x, y = Variable(x.cuda()), y.cuda()
        model.hidden = model.init_hidden()

        output = model(x.t())
        loss = loss_function(output, Variable(y))
        _, predicted = torch.max(output.data, 1)
        valid_preds += predicted.tolist()
        valid_truth += y.tolist()
        valid_loss.append(loss.data[0])
    print("\n**validation epoch %s\
            loss: %.4f, f1:%.4f" % (epoch,
                                    np.mean(valid_loss),
                                    f1_score(valid_truth, valid_preds, average="weighted")))
    print(classification_report(valid_truth, valid_preds,
                                target_names=valid_dataset.cluster_emoji))

