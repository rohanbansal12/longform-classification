import csv
from tqdm import tqdm
import sys
from transformers import BertTokenizer
from joblib import Parallel, delayed

tokenizer = BertTokenizer(
    "/scratch/gpfs/altosaar/dat/longform-data/BERT/bert-base-uncased.txt",
    do_lower_case=True,
)
csv.field_size_limit(sys.maxsize)
ifile = open("fake-news.csv", "r")
reader = csv.reader(ifile)
ofile = open("mapped-fake-news.csv", "w")
writer = csv.writer(ofile, delimiter=",")


def generate_rows(row):
    if len(row) != 17:
        return []
    current_row = []
    current_row.append(row[9])
    raw_text = row[5]
    converted_text = [
        token for token in tokenizer.tokenize(raw_text) if "##" not in token
    ]
    id_text = tokenizer.convert_tokens_to_ids(converted_text)
    current_row.append(id_text)
    current_row.append(8000000)
    current_row.append(row[4])
    current_row.append("fake-news-corpus")
    current_row.append(25)


header_row = ["title", "text", "url", "link", "publication", "model_publication"]
list_of_rows = Parallel(n_jobs=-1, verbose=3)(
    delayed(generate_rows)(row) for row in reader
)
list_of_rows = [row for row in list_of_rows if row]
list_of_rows.pop(0)
list_of_rows.insert(header_row, 0)
writer.writerows(list_of_rows)

