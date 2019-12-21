#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import sys
import argparse
from tqdm import trange
from torchtext import data as torchtext_data
from torchtext import datasets

import torch
import torch.utils.data as data

from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import torch.optim
import torch.nn.functional as F
import numpy as np
from IPython import embed
from operator import add
from run_gpt2 import top_k_logits
from style_utils import to_var
import copy
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim

torch.manual_seed(0)
np.random.seed(0)

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from torch.autograd import Variable

tokenizer = GPT2Tokenizer.from_pretrained('gpt-2_pt_models/345M/')

model = GPT2LMHeadModel.from_pretrained('gpt-2_pt_models/345M/')


class ClassificationHead(torch.nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, class_size=5, embed_size=2048):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = (torch.nn.Linear(embed_size, class_size))

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        # lm_logits = F.relu(self.mlp1(hidden_state))
        # lm_logits = self.mlp2(lm_logits)
        lm_logits = self.mlp(hidden_state)
        return lm_logits


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.classifierhead = ClassificationHead()
        self.model = model
        self.spltoken = Variable(torch.randn(1, 1, 1024).type(torch.FloatTensor), requires_grad=True)
        self.spltoken = self.spltoken.repeat(10, 1, 1)
        self.spltoken = self.spltoken.cuda()

    def train(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass

    def forward(self, x):
        x = model.forward_embed(x)
        x = torch.cat((x, self.spltoken), dim=1)
        _, x = model.forward_transformer_embed(x, add_one=True)
        x = self.classifierhead(x[-1][:, -1, :])
        x = F.log_softmax(x, dim=-1)
        return x


class Discriminator2(torch.nn.Module):
    def __init__(self, class_size=5, embed_size=1024):
        super(Discriminator2, self).__init__()
        self.classifierhead = ClassificationHead(class_size=class_size, embed_size=embed_size)
        self.model = model
        self.embed_size = embed_size

    def get_classifier(self):
        return self.classifierhead

    def train_custom(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass
        self.classifierhead.train()

    def forward(self, x):
        x = model.forward_embed(x)
        hidden, x = model.forward_transformer_embed(x)
        x = torch.sum(hidden, dim=1)
        x = self.classifierhead(x)
        x = F.log_softmax(x, dim=-1)
        return x

class Discriminator2mean(torch.nn.Module):
    def __init__(self, class_size=5, embed_size=1024):
        super(Discriminator2mean, self).__init__()
        self.classifierhead = ClassificationHead(class_size=class_size, embed_size=embed_size)
        self.model = model
        self.embed_size = embed_size

    def get_classifier(self):
        return self.classifierhead

    def train_custom(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass
        self.classifierhead.train()

    def forward(self, x):
        mask_src = 1 - x.eq(0).unsqueeze(1).type(torch.FloatTensor).cuda().detach()
        mask_src = mask_src.repeat(1, self.embed_size, 1)
        x = model.forward_embed(x)
        hidden, x = model.forward_transformer_embed(x)
        #  Hidden has shape batch_size x length x embed-dim

        hidden = hidden.permute(0, 2, 1)
        _, _, batch_length = hidden.shape
        hidden = hidden * mask_src  # / torch.sum(mask_src, dim=-1).unsqueeze(2).repeat(1, 1, batch_length)
        #
        hidden = hidden.permute(0, 2, 1)
        x = torch.sum(hidden, dim=1)/(torch.sum(mask_src, dim=-1).detach() + 1e-10)
        x = self.classifierhead(x)
        x = F.log_softmax(x, dim=-1)
        return x

class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        d = {}
        d['X'] = self.X[index]
        d['y'] = self.y[index]
        return d


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_seqs = torch.zeros(len(sequences), max(lengths)).long().cuda()  # padding index 0
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["X"]), reverse=True)  # sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # input
    x_batch, _ = merge(item_info['X'])
    y_batch = item_info['y']

    return x_batch, torch.tensor(y_batch, device='cuda', dtype=torch.long)


def train_epoch(data_loader, discriminator, device='cuda', args=None, epoch=1):
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator.train_custom()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = discriminator(data)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Relu Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(data_loader, discriminator, device='cuda', args=None):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = discriminator(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\nRelu Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='Train a discriminator on top of GPT-2 representations')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of training epochs')
    parser.add_argument('--save-model', action='store_true', help='whether to save the model')
    parser.add_argument('--dataset-label', type=str, default='SST',choices=('SST', 'clickbait', 'toxic'))
    args = parser.parse_args()

    batch_size = args.batch_size
    device = 'cuda'
    # load sst
    if args.dataset_label == 'SST':
        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(text, label, fine_grained=True, train_subtrees=True,
                                                              # filter_pred=lambda ex: ex.label != 'neutral'
                                                              )
        x = []
        y = []
        d = {"positive": 0, "negative": 1, "very positive": 2, "very negative": 3, "neutral": 4}

        for i in range(len(train_data)):
            seq = TreebankWordDetokenizer().detokenize(vars(train_data[i])["text"])
            seq = tokenizer.encode(seq)
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(d[vars(train_data[i])["label"]])

        dataset = Dataset(x, y)

        test_x = []
        test_y = []
        for i in range(len(test_data)):
            seq = TreebankWordDetokenizer().detokenize(vars(test_data[i])["text"])
            seq = tokenizer.encode(seq)
            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(d[vars(test_data[i])["label"]])
        test_dataset = Dataset(test_x, test_y)
        discriminator = Discriminator2mean(class_size=5).to(device)

    elif args.dataset_label == 'clickbait':
        # data = pickle.load(open("/home/gilocal/lab/exp/language/datasets/clickbait/clickbait.p", "r"))
        with open("datasets/clickbait/clickbait_train_prefix.txt") as f:
            data = []
            for d in f:
                try:
                    data.append(eval(d))
                except:
                    continue
        x = []
        y = []
        for d in data:
            try:
                # seq = tokenizer.encode("Apple's iOS 9 'App thinning' feature will give your phone's storage a boost")
                try:
                    seq = tokenizer.encode(d["text"])
                except:
                    continue
                seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
                x.append(seq)
                y.append(d['label'])
            except:
                pass

        dataset = Dataset(x, y)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        discriminator = Discriminator2mean(class_size=2).to(device)

    elif args.dataset_label == 'toxic':
        # data = pickle.load(open("/home/gilocal/lab/exp/language/datasets/clickbait/clickbait.p", "r"))
        with open("datasets/toxic/toxic_train.txt") as f:
            data = []
            for d in f:
                data.append(eval(d))

        x = []
        y = []
        for d in data:
            try:
                # seq = tokenizer.encode("Apple's iOS 9 'App thinning' feature will give your phone's storage a boost")
                seq = tokenizer.encode(d["text"])

                device = 'cuda'
                if(len(seq)<100):
                    seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
                else:
                    continue
                x.append(seq)
                y.append(int(np.sum(d['label'])>0))
            except:
                pass

        dataset = Dataset(x, y)
        print(dataset)
        print(len(dataset))
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        discriminator = Discriminator2mean(class_size=2).to(device)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        train_epoch(discriminator=discriminator, data_loader=data_loader, args=args, device=device, epoch=epoch)
        test_epoch(data_loader=test_loader, discriminator=discriminator, args=args)
        seq = tokenizer.encode("This is incredible! I love it, this is the best chicken I have ever had.")
        seq = torch.tensor([seq], device=device, dtype=torch.long)
        print(discriminator(seq))

        if (args.save_model):
            torch.save(discriminator.state_dict(),
                       "discrim_models/{}_mean_lin_discriminator_{}.pt".format(args.dataset_label, epoch))
            torch.save(discriminator.get_classifier().state_dict(),
                       "discrim_models/{}_classifierhead.pt".format(args.dataset_label))

    seq = tokenizer.encode("This is incredible! I love it, this is the best chicken I have ever had.")
    seq = torch.tensor([seq], device=device, dtype=torch.long)
    print(discriminator(seq))


if __name__ == '__main__':
    main()

