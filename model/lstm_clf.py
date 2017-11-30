import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, vocab_size, target_size, batch_size,
                 num_layers=1, pretrained_embedding=None, use_gpu=True, embedding_dim=300):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is None:
            print("using pretrained embedding")
            assert pretrained_embedding.shape[0] == vocab_size
            assert pretrained_embedding.shape[1] == embedding_dim
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True)

        self.output = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2))
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        return F.log_softmax(self.output(lstm_out[-1]))
