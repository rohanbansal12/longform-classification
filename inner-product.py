#!/usr/bin/env python
# coding: utf-8

#import necessary libraries
import pandas as pd
import re
import torch
import collections
import numpy as np
import json
import time
import torch.nn as nn
import os
import argparse
import arguments.add_arguments as arguments
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#set and get arguments
parser = argparse.ArgumentParser(description='Train model on article data and test evaluation')
arguments.add_data(parser)
arguments.add_training(parser)
arguments.add_model(parser)
arguments.add_optimization(parser)
args = parser.parse_args()

if not args.create_dicts and args.dict_dir is None:
    parser.error("If --create_dicts is false, --dict_dir must be specified.")

if not args.train_model and args.model_path is None:
    parser.error("If --train_model is false, --model_path must be specified")

#set device
if torch.cuda.is_available() and args.use_gpu:
    device= "cuda"
elif not args.use_gpu:
    device = "cpu"
else:
    print("Cannot use GPU. Using CPU instead.")
    device = "cpu"
print(f"Device: {device}")

#set output directory path
output_path = Path(args.output_dir)

#Tensboard log and graph output folder declaration
log_tensorboard_dir = output_path / "runs" / args.word_embedding_type
writer = SummaryWriter(log_tensorboard_dir)

#define Articles dataset class for easy sampling, iteration, and weight creating
class Articles(torch.utils.data.Dataset):
    def __init__(self, json_file):
        super().__init__()
        with open(json_file, "r") as data_file:
            self.examples = json.loads(data_file.read())

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def tokenize(self):
        for idx, example in enumerate(self.examples):
            self.examples[idx]['text'] = re.findall('[\w]+', self.examples[idx]['text'].lower())
            self.examples[idx]['title'] = re.findall('[\w]+', self.examples[idx]['title'].lower())

    def create_positive_sampler(self):
        prob = np.zeros(len(self))
        for idx, example in enumerate(self.examples):
            if example['model_publication'] == args.target_publication:
                prob[idx] = 1
        return torch.utils.data.WeightedRandomSampler(weights=prob, num_samples=len(self), replacement=True)

    def create_negative_sampler(self):
        prob = np.zeros(len(self))
        for idx, example in enumerate(self.examples):
            if example['model_publication'] != args.target_publication:
                prob[idx] = 1
        return torch.utils.data.WeightedRandomSampler(weights=prob, num_samples=len(self), replacement=True)

    def map_items(self, word_to_id, url_to_id, publication_to_id):
        for idx, example in enumerate(self.examples):
            self.examples[idx]['text'] = [word_to_id.get(word, len(word_to_id)) for word in example['text']]
            self.examples[idx]['text'] = [word for word in example['text'] if word != len(word_to_id)]
            self.examples[idx]['title'] = [word_to_id.get(word, len(word_to_id)) for word in example['title']]
            self.examples[idx]['title'] = [word for word in example['title'] if word != len(word_to_id)]
            self.examples[idx]['url'] = url_to_id.get(example['url'], url_to_id.get("miscellaneous"))
            self.examples[idx]['model_publication'] = publication_to_id.get(example['model_publication'], publication_to_id.get("miscellaneous"))

#function to create dictionaries for words and urls for all datasets at once
def create_merged_dictionaries(all_examples):
    counter = collections.Counter()
    url_counter = collections.Counter()
    publication_counter = collections.Counter()
    urls = []
    publications = ["target"]

    for example in all_examples:
        counter.update(example['text'])
        counter.update(example['title'])
        urls.append(example['url'])
        publications.append(example['model_publication'])

    url_counter.update(urls)
    publication_counter.update(publications)
    word_to_id = {word: id for id, word in enumerate(counter.keys())}
    article_to_id = {word: id for id, word in enumerate(url_counter.keys())}
    publication_to_id = {publication: id for id, publication in enumerate(publication_counter.keys())}
    word_to_id.update({"miscellaneous":len(word_to_id)})
    article_to_id.update({"miscellaneous":len(article_to_id)})
    publication_to_id.update({"miscellaneous":len(publication_to_id)})
    return word_to_id, article_to_id, publication_to_id

#load datasets
train_path = Path(args.train_path)
test_path = Path(args.test_path)
eval_path = Path(args.eval_path)

train_data = Articles(train_path)
test_data = Articles(test_path)
eval_data = Articles(eval_path)
print("Data Loaded")

#Check if items need to be tokenized
if args.map_items and args.tokenize:
    train_data.tokenize()
    test_data.tokenize()
    eval_data.tokenize()
    print("Items tokenized")

#Create and save or load dictionaries based on arguments
if args.create_dicts:
    all_examples = train_data.examples+test_data.examples+eval_data.examples
    final_word_ids,final_url_ids, final_publication_ids = create_merged_dictionaries(all_examples)
    print("Dictionaries Created")
    dict_path = Path(args.data_dir) / "dictionaries"
    if not dict_path.is_dir():
        dict_path.mkdir()

    #save dictionary files for future use and ease of access
    word_dict_path = dict_path / "word_dictionary.json"
    article_dict_path = dict_path / "article_dictionary.json"
    publication_dict_path = dict_path / "publication_dictionary.json"
    with open(word_dict_path, "w") as file:
        json.dump(final_word_ids, file)

    with open(article_dict_path, "w") as file:
        json.dump(final_url_ids, file)

    with open(publication_dict_path, "w") as file:
        json.dump(final_publication_ids, file)
    print("Dictionaries saved to /dictionary folder.")

else:
    abs_dictionary_dir = Path(args.dict_dir)
    word_dict_path = abs_dictionary_dir / "word_dictionary.json"
    url_id_path = abs_dictionary_dir / "article_dictionary.json"
    publication_id_path = abs_dictionary_dir / "publication_dictionary.json"

    if Path(word_dict_path).is_file() and Path(url_id_path).is_file() and Path(publication_id_path).is_file():
        with open(word_dict_path, "r") as file:
            final_word_ids = json.load(file)

        with open(url_id_path, "r") as file:
            final_url_ids = json.load(file)

        with open(publication_id_path, "r") as file:
            final_publication_ids = json.load(file)

        print("Dictionaries Loaded")
    else:
        parser.error("Necessary files word_dictionary.json, article_dictionary.json and publication_dictionary.json not found in --dict_dir.")

#map items in dataset using dictionary keys (convert words and urls to numbers for the model)
if args.map_items:
    for dataset in [train_data, test_data, eval_data]:
        dataset.map_items(final_word_ids, final_url_ids, final_publication_ids)
    print("Items mapped")
    mapped_data_path = Path(args.data_dir) / "mapped-data"
    if not mapped_data_path.is_dir():
        mapped_data_path.mkdir()

    train_mapped_path = mapped_data_path / "train.json"
    test_mapped_path = mapped_data_path / "test.json"
    eval_mapped_path = mapped_data_path / "evaluation.json"
    with open(train_mapped_path, "w") as file:
        json.dump(train_data.examples, file)
    with open(test_mapped_path, "w") as file:
        json.dump(test_data.examples, file)
    with open(eval_mapped_path, "w") as file:
        json.dump(eval_data.examples, file)
    print(f"Mapped Data saved to {mapped_data_path} directory")

#define model which uses a simple dot product with publication and word embeddings to calculate logits
class InnerProduct(nn.Module):
    def __init__(self, n_publications, n_articles, n_attributes, emb_size, sparse, use_article_emb, mode):
        super().__init__()
        self.emb_size = emb_size
        self.publication_embeddings = nn.Embedding(n_publications, emb_size, sparse=sparse)
        self.publication_bias = nn.Embedding(n_publications, 1, sparse=sparse)
        self.attribute_emb_sum = nn.EmbeddingBag(n_attributes, emb_size, mode=mode, sparse=sparse)
        self.attribute_bias_sum = nn.EmbeddingBag(n_attributes, 1, mode=mode, sparse=sparse)
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


#Create batches with positive samples in first half and negative examples in second half
class BatchSamplerWithNegativeSamples(torch.utils.data.Sampler):
    def __init__(self, pos_sampler, neg_sampler, batch_size, items):
        self._pos_sampler = pos_sampler
        self._neg_sampler = neg_sampler
        self._items = items
        assert batch_size % 2 == 0, 'Batch size must be divisible by two for negative samples.'
        self._batch_size = batch_size

    def __iter__(self):
        batch, neg_batch = [], []
        neg_sampler = iter(self._neg_sampler)
        for pos_idx in self._pos_sampler:
            batch.append(pos_idx)
            neg_idx = pos_idx
            # keep sampling until we get a true negative sample
            while self._items[neg_idx] == self._items[pos_idx]:
                try:
                    neg_idx = next(neg_sampler)
                except StopIteration:
                    neg_sampler = iter(self._neg_sampler)
                    neg_idx = next(neg_sampler)
            neg_batch.append(neg_idx)
            if len(batch) == self._batch_size // 2:
                batch.extend(neg_batch)
                yield batch
                batch, neg_batch = [], []
        return

    def __len__(self):
        return len(self._pos_sampler) // self._batch_size

#define function to return necessary data for dataloader to pass into model
def collate_fn(examples):
    words = []
    articles = []
    labels = []
    publications = []
    for example in examples:
        if args.use_all_words:
            words.append(list(set(example['text'])))
        else:
            if len(example['text']) > args.words_to_use:
                words.append(list(set(example['text'][:args.words_to_use])))
            else:
                words.append(list(set(example['text'])))
        articles.append(example['url'])
        publications.append(example['model_publication'])
        labels.append(example['model_publication'])
    num_words = [len(x) for x in words]
    words = np.concatenate(words, axis=0)
    word_attributes = torch.tensor(words, dtype=torch.long)
    articles = torch.tensor(articles, dtype=torch.long)
    num_words.insert(0,0)
    num_words.pop(-1)
    attribute_offsets = torch.tensor(np.cumsum(num_words), dtype=torch.long)
    publications = torch.tensor(publications, dtype=torch.long)
    real_labels = torch.tensor(labels, dtype=torch.long)
    return publications, articles, word_attributes, attribute_offsets, real_labels

#change negative example publication ids to the ids of the first half for predictions
def collate_with_neg_fn(examples):
    publications, articles, word_attributes, attribute_offsets, real_labels = collate_fn(examples)
    publications[len(publications)//2:] = publications[:len(publications)//2]
    return publications, articles, word_attributes, attribute_offsets, real_labels

#create weights for dataset samples to ensure only positive and negative examples are chosen in respective samples
pos_sampler = train_data.create_positive_sampler()
neg_sampler = train_data.create_negative_sampler()

#create batch_sampler with positive samples in first half and negative samples in second half with specific batch size
train_batch_sampler = BatchSamplerWithNegativeSamples(
    pos_sampler=pos_sampler, neg_sampler=neg_sampler,
    items=train_data.examples, batch_size=args.batch_size)

#create dataloaders for iterable data when training and testing recall
if device=="cuda":
    pin_mem = True
else:
    pin_mem = False

train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=train_batch_sampler, collate_fn=collate_with_neg_fn, pin_memory=pin_mem)

eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), collate_fn=collate_fn, pin_memory=pin_mem)

#function that allows for infinite iteration over training batches
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#generate validation batch information prior to running loop as data is all collected in one batch
eval_batch = next(iter(eval_loader))
eval_publications, eval_articles, eval_word_attributes, eval_attribute_offsets, eval_real_labels = eval_batch
eval_articles = eval_articles.to(device)
eval_word_attributes = eval_word_attributes.to(device)
eval_attribute_offsets = eval_attribute_offsets.to(device)
eval_real_labels = eval_real_labels.to(device)

if args.train_model:

    #initialize model, loss, and optimizer
    kwargs = dict(n_publications=len(final_publication_ids),
                  n_articles=len(final_url_ids),
                  n_attributes=len(final_word_ids),
                  emb_size=args.emb_size, sparse=args.use_sparse,
                  use_article_emb=args.use_article_emb,
                  mode=args.word_embedding_type)
    model = InnerProduct(**kwargs)
    model.reset_parameters()
    model.to(device)

    loss = torch.nn.BCEWithLogitsLoss()
    if args.optimizer_type == "RMS":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    print(model)
    print(optimizer)
    model.train() # turn on training mode
    check=True
    running_loss = 0
    print("Beginning Training")
    print("--------------------")
    #training loop with validation checks every 50 steps and final validation recall calculated after 400 steps
    while check:
        for step,batch in enumerate(cycle(train_loader)):
            optimizer.zero_grad();
            publications, articles, word_attributes, attribute_offsets, real_labels = batch
            publications = publications.to(device)
            articles = articles.to(device)
            word_attributes = word_attributes.to(device)
            attribute_offsets = attribute_offsets.to(device)
            labels = torch.Tensor((np.arange(len(articles)) < len(articles) // 2).astype(np.float32)) #create fake labels with first half as positive(1) and second half as negative(0)
            labels = labels.to(device)
            logits = model(publications, articles, word_attributes, attribute_offsets)
            L = loss(logits, labels)
            L.backward()
            optimizer.step()
            running_loss += L.item()
            if step % 100 == 0 and step % args.training_steps != 0:
                writer.add_scalar('Loss/train', running_loss/100, step)
                print(f"Training Loss: {running_loss/100}")
                model.eval();
                publication_set = [args.target_publication]*len(eval_data)
                publication_set = torch.tensor(publication_set, dtype=torch.long)
                publication_set = publication_set.to(device)
                preds = model(publication_set, eval_articles, eval_word_attributes, eval_attribute_offsets)
                sorted_preds, indices = torch.sort(preds, descending=True)
                correct_10=0
                correct_100=0
                for i in range(0, 100):
                    if eval_real_labels[indices[i]] == args.target_publication:
                        if i < 10:
                            correct_10 += 1
                        correct_100 += 1
                print(f"Evaluation Performance: Step - {step}")
                print(f"Top 10: {correct_10} / 10 or {correct_10*10} %")
                print(f"Top 100: {correct_100} / 100 or {correct_100} %")
                print("--------------------")
                writer.add_scalar('Eval/Top-10', correct_10, step)
                writer.add_scalar('Eval/Top-100', correct_100, step)
                model.train();
                running_loss = 0.0
            if step != 0 and step % args.training_steps == 0:
                writer.add_scalar('Loss/train', running_loss/100, step)
                print(f"Training Loss: {running_loss/100}")
                print("Getting Final Evaluation Results")
                print("--------------------")
                model.eval();
                publication_set = [args.target_publication]*len(eval_data)
                publication_set = torch.tensor(publication_set, dtype=torch.long)
                publication_set = publication_set.to(device)
                preds = model(publication_set, eval_articles, eval_word_attributes, eval_attribute_offsets)
                sorted_preds, indices = torch.sort(preds, descending=True)
                correct_10=0
                correct_100=0
                for i in range(0, 100):
                    if eval_real_labels[indices[i]] == args.target_publication:
                        if i < 10:
                            correct_10 += 1
                        correct_100 += 1
                print(f"Evaluation Performance: Step - {step}")
                print(f"Top 10: {correct_10} / 10 or {correct_10*10} %")
                print(f"Top 100: {correct_100} / 100 or {correct_100 }%")
                print("--------------------")
                writer.add_scalar('Eval/Top-10', correct_10, step)
                writer.add_scalar('Eval/Top-100', correct_100, step)
                model.train();
                writer.close()
                df = pd.DataFrame(columns=['title', 'url', 'text','publication', 'target_prediction'])
                links = list(final_url_ids.keys())
                for i in range(0, 1500):
                    example = eval_data[indices[i]]
                    prediction = sorted_preds[i].item()
                    text = []
                    for x in example['title']:
                        text.append(next((word for word, numero in final_word_ids.items() if numero == x), None))
                        title = ""
                    for word in text:
                        title += word
                        title += " "
                    unique_text = list(set(example['text']))
                    url = links[example['url']]
                    publication = example['publication']
                    df.loc[i] = [title, url, unique_text, publication, prediction]
                results_path = output_path / "results"
                if not results_path.is_dir():
                    results_path.mkdir()
                evaluation_results_path = results_path / "evaluation"
                if not evaluation_results_path.is_dir():
                    evaluation_results_path.mkdir()
                result_path = args.word_embedding_type + "-top-1500.csv"
                eval_folder_path = evaluation_results_path / result_path
                df.to_csv(eval_folder_path, index=False)
                eval_numeric_path = evaluation_results_path / "numeric_results.txt"
                with eval_numeric_path.open(mode="w+") as file:
                    file.writelines(f"Top 10: {correct_10} / 10 or {correct_10*10} %" + os.linesep)
                    file.writelines(f"Top 100: {correct_100} / 100 or {correct_100 }%")
                check=False
                break

    #save model for easy future reloading
    model_path = output_path / "model"
    if not model_path.is_dir():
        model_path.mkdir()
    model_string = args.word_embedding_type + "-inner-product-model.pt"
    model_path = model_path / model_string
    torch.save(model.state_dict(), model_path)

else:
    abs_model_path = Path(args.model_path)
    kwargs = dict(n_publications=len(final_publication_ids),
              n_articles=len(final_url_ids),
              n_attributes=len(final_word_ids),
              emb_size=args.emb_size, sparse=args.use_sparse,
              use_article_emb=args.use_article_emb,
              mode=args.word_embedding_type)
    model = InnerProduct(**kwargs)
    model.load_state_dict(torch.load(abs_model_path))
    print(model)
    print("Model Successfully Loaded")
    model.to(device)
    print("Getting Final Evaluation Results")
    print("--------------------")
    model.eval();
    publication_set = [args.target_publication]*len(eval_data)
    publication_set = torch.tensor(publication_set, dtype=torch.long)
    publication_set = publication_set.to(device)
    preds = model(publication_set, eval_articles, eval_word_attributes, eval_attribute_offsets)
    sorted_preds, indices = torch.sort(preds, descending=True)
    correct_10=0
    correct_100=0
    for i in range(0, 100):
        if eval_real_labels[indices[i]] == args.target_publication:
            if i < 10:
                correct_10 += 1
            correct_100 += 1
    print("Evaluation Performance:")
    print("Top 10: ", correct_10, "/10 or ", (correct_10*10), "%")
    print("Top 100: ", correct_100, "/100 or", correct_100, "%")
    print("--------------------")
    df = pd.DataFrame(columns=['title', 'url', 'text','publication', 'target_prediction'])
    links = list(final_url_ids.keys())
    for i in range(0, 1500):
        example = eval_data[indices[i]]
        prediction = sorted_preds[i].item()
        text = []
        for x in example['title']:
            text.append(next((word for word, numero in final_word_ids.items() if numero == x), None))
            title = ""
        for word in text:
            title += word
            title += " "
        unique_text = list(set(example['text']))
        url = links[example['url']]
        publication = example['publication']
        df.loc[i] = [title, url, unique_text, publication, prediction]
    results_path = output_path / "results"
    if not results_path.is_dir():
        results_path.mkdir()
    evaluation_results_path = results_path / "evaluation"
    if not evaluation_results_path.is_dir():
        evaluation_results_path.mkdir()
    result_path = args.word_embedding_type + "-top-1500.csv"
    eval_folder_path = evaluation_results_path / result_path
    df.to_csv(eval_folder_path, index=False)
    eval_numeric_path = evaluation_results_path / "numeric_results.txt"
    with eval_numeric_path.open(mode="w+") as file:
        file.writelines(f"Top 10: {correct_10} / 10 or {correct_10*10} %" + os.linesep)
        file.writelines(f"Top 100: {correct_100} / 100 or {correct_100 }%")
