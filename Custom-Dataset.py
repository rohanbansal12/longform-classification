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
batch_sampler = train.create_weighted_sampler()
loader = torch.utils.data.DataLoader(train.examples, batch_sampler=batch_sampler, collate_fn=collate_fn)
next(iter(loader))
