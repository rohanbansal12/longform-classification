from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
import pandas as pd

evalu = pd.read_json("../../../../data/final-data/evaluation.json")
test = pd.read_json("../../../../data/final-data/test.json")
train = pd.read_json("../../../../data/final-data/train.json")

tokenizer = BertWordPieceTokenizer("bert-base-uncased.txt", lowercase=True)


def tokenize_text(text):
    ids = tokenizer.encode(text).ids
    ids.pop()
    ids.pop(0)
    return ids


evalu.text.apply(tokenize_text)

test.text.apply(tokenize_text)
train.text.apply(tokenize_text)

for dataset in [train, evalu, test]:
    dataset = dataset[~dataset.link.str.contains("www.businessweek")]
    dataset = dataset[~dataset.link.str.contains("www.bloomberg")]
    dataset.reset_index(drop=True)

evalu.to_json("mapped-data/evaluation.json")
train.to_json("mapped-data/train.json")
test.to_json("mapped-data/test.json")
