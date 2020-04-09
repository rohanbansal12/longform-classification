#import necessary libraries
from torchtext.data import Field
from torchtext.data import TabularDataset
import pandas as pd
import re
import torch
import collections
import numpy as np
from torchtext.data import Iterator, BucketIterator

#output all items, not just last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
  
def tokenize(text):
    return re.findall('[\w]+', text.lower())

TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
URL = Field(sequential=False, lower=True, tokenize=None)
LABEL = Field(sequential=False, use_vocab=False)

datafields = {'title':('title', TEXT), "text":('text', TEXT), "url":("url", URL), "longform":("longform", LABEL)}

train,test,val = TabularDataset.splits(path="changed-data/debugdata", 
                                        train="train_basic.json", 
                                          test="test_basic.json", 
                                          validation="val_basic.json", 
                                          format="json", 
                                          fields = datafields)

counter = collections.Counter()
url_counter = collections.Counter()
urls = []

for example in train.examples+test.examples+val.examples:
    counter.update(example.text)
    counter.update(example.title)
    urls.append(example.url)

url_counter.update(urls)
word_to_id = {word: id for id, word in enumerate(counter.keys())}
article_to_id = {word: id for id, word in enumerate(url_counter.keys())}

def create_weighted_sampler(dataset):
    prob = np.zeros(len(dataset), dtype=np.float32)
    positive = sum(example.longform == 0 for example in dataset.examples)
    negative = len(dataset) - positive
    for idx, example in enumerate(dataset.examples):
        if example.longform == 0:
            prob[idx] = (positive/(len(dataset)))
        else:
            prob[idx] = (negative/(len(dataset)))
    return torch.utils.data.WeightedRandomSampler(weights=prob, num_samples=len(dataset))

def collate_fn(examples):
    words = []
    articles = []
    for example in examples:
        words.append(example.text)
        articles.append(example.url)
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
    return publications, articles, word_attributes, attribute_offsets
    
train_data = torch.utils.data.DataLoader(train, 
                                        batch_sampler=create_weighted_sampler(train), 
                                        collate_fn=collate_fn)
                                        
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

loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.Adam(model.parameters(), lr=1e-2)

for step, batch in enumerate(iter(util.cycle(train_data))):
    model.zero_grad()
    logits = model(*[x.to("cpu") for x in batch])
    L = loss(logits, labels)
    L.backward()
    optimzer.step()
