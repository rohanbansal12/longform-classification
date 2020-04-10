#import necessary libraries
from torchtext.data import Field
from torchtext.data import TabularDataset
import pandas as pd
import re
import torch
import torch.nn as nn
import collections
import numpy as np
from torchtext.data import Iterator, BucketIterator
import json
import time
import tqdm

#output all items from cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#set device
if torch.cuda.is_available():
    device= "cuda"
else:
    device = "cpu"  

#Custom Articles Class with method definitions
class Articles(torch.utils.data.Dataset):
    def __init__(self, json_file):
        super().__init__()
        with open(json_file, "r") as data_file:
            self.examples = json.loads(data_file.read())
        self.tokenize()
    
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
    
    def create_dictionaries(self):
        counter = collections.Counter()
        url_counter = collections.Counter()
        urls = []

        for example in self.examples:
            counter.update(example['text'])
            counter.update(example['title'])
            urls.append(example['url'])

        url_counter.update(urls)
        word_to_id = {word: id for id, word in enumerate(counter.keys())}
        article_to_id = {word: id for id, word in enumerate(url_counter.keys())}
        return word_to_id, article_to_id
    
    def map_items(self, word_to_id, url_to_id):
        words = []
        articles = []
        labels = []
        for idx, example in enumerate(self.examples):
            self.examples[idx]['text'] = [word_to_id.get(word) for word in example['text']]
            self.examples[idx]['title'] = [word_to_id.get(word) for word in example['title']]
            self.examples[idx]['url'] = url_to_id.get(example['url'])

#merge dictionaries for different sets
def create_merged_dictionaries(train, test, val):
    train_word_id, train_url_id = train.create_dictionaries()
    test_word_id, test_url_id = test.create_dictionaries()
    val_word_id, val_url_id = val.create_dictionaries()
    test_word_id.update(train_word_id)
    test_url_id.update(train_url_id)
    val_word_id.update(test_word_id)
    val_url_id.update(test_url_id)
    return val_word_id, val_url_id

#load data
train_data = Articles("changed-data/train.json")
test_data = Articles("changed-data/test.json")
val_data = Articles("changed-data/validate.json")

#create dictionaries from loaded data
final_word_ids,final_url_ids= create_merged_dictionaries(train_data, test_data, val_data)

#convert fields to their mapping for all datasets
for dataset in [train_data, test_data, val_data]:
    dataset.map_items(final_word_ids, final_url_ids);

#save dictionaries
with open("dictionaries/word_dictionary.json", "w") as file:
    json.dump(final_word_ids, file)

with open("dictionaries/article_dictionary.json", "w") as file:
    json.dump(final_url_ids, file)

#function used in DataLoader to output labels for batches
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
    word_attributes = torch.tensor(words, dtype=torch.long)
    articles = torch.tensor(articles, dtype=torch.long)
    num_words.insert(0,0)
    num_words.pop(-1)
    attribute_offsets = torch.tensor(np.cumsum(num_words), dtype=torch.long)
    publications = torch.tensor([0])
    labels = torch.tensor(labels, dtype=torch.float)
    return publications, articles, word_attributes, attribute_offsets, labels

#model definition using dot product
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
        
#assign weights to train set to ensure even batches       
train_weight_sampler = train_data.create_weighted_sampler()

#wrap with a BatchSampler to create batches of size 64
train_batch_sampler = torch.utils.data.BatchSampler(train_weight_sampler, 64, drop_last=True)

#Create iterable batches of data points
train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=train_batch_sampler, collate_fn=collate_fn, pin_memory=True)

#Create batch with entire validation set
val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), collate_fn=collate_fn, pin_memory=True)

#initialize model, loss, and optimizer
kwargs = dict(n_publications=1, 
              n_articles=len(final_url_ids), 
              n_attributes=len(final_word_ids), 
              emb_size=100, sparse=False, 
              use_article_emb=True)
model = InnerProduct(**kwargs)
model.reset_parameters()
model.to(device)

loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4,momentum=0.9)

#get validation data for validation recall
val_batch = next(iter(val_loader))
val_publications, val_articles, val_word_attributes, val_attribute_offsets, val_labels = val_batch
val_publications = val_publications.to(device)
val_articles = val_articles.to(device)
val_word_attributes = val_word_attributes.to(device)
val_attribute_offsets = val_attribute_offsets.to(device)
val_labels = val_labels.to(device)

#function to keep the train_loader going infinitely
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
model.train() # turn on training mode

#start training on train set and test every 50 steps on validation
while True: 
    for step,batch in enumerate(tqdm(cycle(train_loader))):
        optimizer.zero_grad()
        publications, articles, word_attributes, attribute_offsets, labels = batch
        publications = publications.to(device)
        articles = articles.to(device)
        word_attributes = word_attributes.to(device)
        attribute_offsets = attribute_offsets.to(device)
        labels = labels.to(device)
        logits = model(publications, articles, word_attributes, attribute_offsets)
        L = loss(logits, labels)
        L.backward()
        optimizer.step()
        if step % 50 == 0:
            model.eval()
            preds = model(val_publications, val_articles, val_word_attributes, val_attribute_offsets)
            sorted_preds, indices = torch.sort(preds, descending=True)
            correct_10=0
            correct_100=0
            for i in range(0,10):
                if val_labels[indices[i]] == 1:
                    correct_10 += 1
            for i in range(0, 100):
                if val_labels[indices[i]] == 1:
                    correct_100 += 1
            print("Top 10: ", correct_10, " /10 or ", (correct_10*10), "%")
            print("Top 100: ", correct_100, " /100 or ", correct_100, "%")
