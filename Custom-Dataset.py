#import necessary libraries
from torchtext.data import Field
from torchtext.data import TabularDataset
import pandas as pd
import re
import torch
import collections
import numpy as np
from torchtext.data import Iterator, BucketIterator
import json
import torch.nn as nn
import tqdm.notebook as tqdm

#output all items, not just last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#set device
device = "cpu"  

class Articles(torch.utils.data.Dataset):
    def __init__(self, json_file):
        super().__init__()
        with open(json_file, "r") as data_file:
            self.examples = json.loads(data_file.read())
        self.tokenize()
        print(self.examples[0])
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)
    
    def tokenize(self):
        for idx, example in enumerate(self.examples):
            self.examples[idx]['text'] = re.findall('[\w]+', self.examples[idx]['text'].lower())
            self.examples[idx]['title'] = re.findall('[\w]+', self.examples[idx]['title'].lower())
    
    def create_weighted_sampler(self):
        prob = np.zeros(len(self))
        positive = sum(example['longform'] == 0 for example in self.examples)
        negative = len(self) - positive
        for idx, example in enumerate(self.examples):
            if example['longform'] == 0:
                prob[idx] = (positive/(len(self)))
            else:
                prob[idx] = (negative/(len(self)))
        return torch.utils.data.WeightedRandomSampler(weights=prob, num_samples=len(self), replacement=True)
    
def collate_fn(examples):
    words = []
    articles = []
    labels = []
    for example in examples:
        words.append(example['text'])
        articles.append(example['url'])
        labels.append(example['longform'])
    num_words = [len(x) for x in words]
    words = np.concatenate(words, axis=0)
    word_attributes = [word_to_id.get(word) for word in words]
    word_attributes = torch.tensor(word_attributes, dtype=torch.long)
    articles = [article_to_id.get(article) for article in articles]
    articles = torch.tensor(articles, dtype=torch.long)
    num_words.insert(0,0)
    num_words.pop(-1)
    attribute_offsets = torch.tensor(np.cumsum(num_words), dtype=torch.long)
    publications = torch.tensor([0])
    labels = torch.tensor(labels, dtype=torch.long)
    return publications, articles, word_attributes, attribute_offsets, labels
    
train = Articles("changed-data/debugdata/train_basic.json")
weight_sampler = train.create_weighted_sampler()
batch_sampler = torch.utils.data.BatchSampler(weight_sampler, 10, drop_last=True)

word_to_id, article_to_id = train.map_items()
loader = torch.utils.data.DataLoader(train.examples, batch_sampler=batch_sampler, collate_fn=collate_fn)
next(iter(loader))

class InnerProduct(nn.Module):
    def __init__(self, n_publications, n_articles, n_attributes, emb_size, sparse, use_article_emb):
        super().__init__()
        self.emb_size = emb_size
        self.publication_embeddings = nn.Embedding(n_publications, emb_size, sparse=sparse)
        self.publication_bias = nn.Embedding(n_publications, 1, sparse=sparse)
        self.attribute_emb_sum = nn.EmbeddingBag(n_attributes, emb_size, mode='sum', sparse=sparse)
        self.attribute_bias_sum = nn.EmbeddingBag(n_attributes, 1, mode='sum', sparse=sparse)
        self.use_article_emb = use_article_emb
        if use_article_emb:
            self.article_embeddings = nn.Embedding(n_articles, emb_size, sparse=sparse)
            self.article_bias = nn.Embedding(n_articles, 1, sparse=sparse)
        self.use_article_emb = use_article_emb

    def reset_parameters(self):
        for module in [self.publication_embeddings, self.attribute_emb_sum]:
            scale = 0.07
            nn.init.uniform_(module.weight, -scale, scale)
        for module in [self.publication_bias, self.attribute_bias_sum]:
            nn.init.zeros_(module.weight)
        if self.use_article_emb:
            for module in [self.article_embeddings, self.article_bias]:
            # initializing article embeddings to zero to allow large batch sizes
            # nn.init.uniform_(module.weight, -scale, scale)
                nn.init.zeros_(module.weight)

    def forward(self, publications, articles, word_attributes, attribute_offsets, pairwise=False, return_intermediate=False):
        publication_emb = self.publication_embeddings(publications)
        attribute_emb = self.attribute_emb_sum(word_attributes, attribute_offsets)
        if self.use_article_emb:
            article_and_attr_emb = self.article_embeddings(articles) + attribute_emb
        else:
            article_and_attr_emb = attribute_emb
        attr_bias = self.attribute_bias_sum(word_attributes, attribute_offsets)
        publication_bias = self.publication_bias(publications)
        if pairwise:
          # for every publication, compute inner product with every article
          # (publications, emb_size) x (emb_size, articles) -> (publications, articles)
            inner_prod = publication_emb @ article_and_attr_emb.t()
          # broadcasting across publication dimension
            logits = inner_prod + publication_bias
          # broadcast across article dimension
            logits += attr_bias.t()
            if self.use_article_emb:
                logits += self.article_bias(articles).t()
        else:
              # for every publication, only compute inner product with corresponding minibatch element
              # (batch_size, 1, emb_size) x (batch_size, emb_size, 1) -> (batch_size, 1)
              # logits = torch.bmm(publication_emb.view(-1, 1, self.emb_size), 
              #                    (article_and_attr_emb).view(-1, self.emb_size, 1)).squeeze()
            inner_prod = (publication_emb * article_and_attr_emb).sum(-1)
            logits = inner_prod + attr_bias.squeeze() + publication_bias.squeeze()
            if self.use_article_emb:
                logits += self.article_bias(articles).squeeze()
        if return_intermediate:
            return logits, publication_emb, attribute_emb
        else:
            return logits
        
kwargs = dict(n_publications=1, 
              n_articles=len(article_to_id), 
              n_attributes=len(word_to_id), 
              emb_size=100, sparse=False, 
              use_article_emb=True)
model = InnerProduct(**kwargs)
model.reset_parameters()

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4,momentum=0.9)
epochs = 2

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train() # turn on training mode
    for step, batch in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        publications, articles, word_attributes, attribute_offsets, labels = batch
        logits = model(publications, articles, word_attributes, attribute_offsets)
        L = loss(logits, labels)
        L.backward()
        optimizer.step()
        running_loss += L.data * len(articles)

    epoch_loss = running_loss / len(train)
    print("Epoch loss: ", epoch_loss)
