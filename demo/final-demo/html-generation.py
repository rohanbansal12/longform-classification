# import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import sys

sys.path.append(
    "/users/rohan/news-classification/ranking-featured-writing/rankfromsets"
)
import os
import argparse
from data_processing.articles import Articles
from models.models import InnerProduct
import data_processing.dictionaries as dictionary
import scipy
import json


# get arguments for script and parse
def expand_path(string):
    return Path(os.path.expandvars(string))


parser = argparse.ArgumentParser(
    description="Generate default website demo html output for a specific dataset"
)
parser.add_argument(
    "--model_matrix_dir",
    type=expand_path,
    required=True,
    help="This is required to load model matrices.",
)

parser.add_argument(
    "--data_matrix_path",
    type=expand_path,
    required=True,
    help="This is required to load dictionaries",
)

parser.add_argument(
    "--dict_dir", type=expand_path, required=True, help="Path to data to be ranked."
)

parser.add_argument(
    "--html_output_dir",
    type=expand_path,
    required=True,
    help="The place to store the generated html.",
)

parser.add_argument(
    "--real_data_path",
    type=expand_path,
    required=True,
    help="Mapped and filtered data to generate html with.",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Indicates which dataset for demo this is.",
)

args = parser.parse_args()

# load dictionaries
dict_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(
    dict_dir
)
print("Dictionaries loaded.")

# load numeric data for calculations
publication_emb = np.asarray(
    [
        1.293169,
        0.5968991,
        0.08624366,
        -1.340886,
        0.20788613,
        -1.8755758,
        -1.048286,
        -0.89399487,
        -0.6858106,
        -0.5503035,
        -0.7068237,
        -0.07774379,
        -0.5711354,
        -0.06891353,
        -0.6157936,
        0.18361174,
        1.0387663,
        -0.39592394,
        -0.01400398,
        0.93364245,
        0.31408283,
        -1.5882636,
        -1.1210498,
        1.3615427,
        0.04305666,
        -0.25838053,
        0.40041822,
        0.64693785,
        1.355679,
        -0.3234386,
        -0.28630123,
        -1.2223343,
        -0.5902191,
        0.31748208,
        0.6941454,
        0.54294777,
        0.61911935,
        -0.50641966,
        -1.1085368,
        -0.06808531,
        -1.9941399,
        1.4491764,
        0.13361451,
        -0.65689725,
        1.430887,
        -0.03157316,
        -0.10601741,
        1.3829597,
        0.18713853,
        -0.7091327,
        -1.0947673,
        0.33397934,
        -0.6568065,
        0.72214764,
        0.07785704,
        0.39983407,
        1.9645799,
        -2.0063865,
        -0.03951475,
        -1.0200714,
        1.6216166,
        -0.32680988,
        1.2766384,
        0.56323916,
        0.7573127,
        0.02096812,
        0.8105754,
        -0.07347149,
        -0.17383198,
        0.78832185,
        1.3038499,
        0.9610421,
        0.9688594,
        0.45888966,
        -0.87990636,
        1.397912,
        -0.87064624,
        0.4131552,
        1.201137,
        0.13622381,
        -1.6119416,
        -0.892891,
        -1.022667,
        0.25875703,
        -0.43832338,
        -0.8434626,
        1.0337343,
        2.521364,
        -2.3373847,
        2.627967,
        -0.5229926,
        0.13273215,
        -1.1643656,
        -1.2185771,
        0.06260775,
        0.24288402,
        -1.0325723,
        -1.638396,
        -1.7829332,
        0.23202017,
    ],
    dtype=np.float32,
)
publication_bias = 3

word_article_path = args.data_matrix_path
word_articles = scipy.sparse.load_npz(word_article_path)

word_emb_path = args.model_matrix_dir / "word_emb.npy"
word_emb = np.load(word_emb_path)
word_bias_path = args.model_matrix_dir / "word_bias.npy"
word_bias = np.load(word_bias_path)

# perform mathematical calculations to get default top predictions
print(word_articles.shape)
print(word_emb.shape)
article_embeddings = word_articles.dot(word_emb)

emb_times_publication = np.dot(article_embeddings, publication_emb.reshape(100, 1))

article_bias = word_articles.dot(word_bias)

product_with_bias = emb_times_publication + article_bias

word_counts = word_articles.sum(axis=1).reshape(word_articles.shape[0], 1)

final_logits = np.divide(product_with_bias, word_counts) + float(publication_bias)

indices = final_logits.argsort(axis=0)[-75:].reshape(75)

word_logits = np.dot(word_emb, publication_emb.reshape(100, 1)) + word_bias

top_articles = word_articles[indices.tolist()[0]]

broadcasted_words_per_article = top_articles.toarray() * word_logits.T

sorted_word_indices = broadcasted_words_per_article.argsort(axis=1)

return_articles = []

raw_data = Articles(args.real_data_path)
id_to_word = {v: k for k, v in final_word_ids.items()}

i = 0
for idx in indices.tolist()[0]:
    current_article = raw_data[int(idx)]
    current_article["logit"] = float(final_logits[int(idx)])
    current_sorted_words = sorted_word_indices[i]
    top_words = []
    least_words = []
    for top_word in current_sorted_words[-20:]:
        word = id_to_word[top_word]
        if "unused" not in word and "##" not in word and len(word) > 1:
            top_words.append(word)
    for least_word in current_sorted_words[:20]:
        word = id_to_word[least_word]
        if "unused" not in word and "##" not in word and len(word) > 1:
            least_words.append(word)
    current_article["top_words"] = top_words
    current_article["least_words"] = least_words
    return_articles.append(current_article)
    i += 1


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# generate default html for the predictions
ordered_return_articles = return_articles[::-1]
grand_html = []

for idx, article in enumerate(ordered_return_articles):
    if not article["publication"] or len(article["publication"]) > 35:
        continue
    grand_html.append("<tr>")
    grand_html.append(f"<td class=\"logit\">{round(article['logit'], 3)}</td>")
    grand_html.append(
        f"<td class=\"title mdl-data-table__cell--non-numeric\"><a class=\"article_link\" href=\"{article['link']}\">{article['title']}</a>"
    )
    grand_html.append("<br></br>")
    grand_html.append(f"<p class=\"publication\">{article['publication']}</p>")
    top_word_list = ""
    for item in article["top_words"]:
        if len(item) > 2 and not RepresentsInt(item):
            top_word_list += item
            top_word_list += ", "
    least_word_list = ""
    for item in article["least_words"]:
        if len(item) > 2:
            least_word_list += item
            least_word_list += ", "
    grand_html.append(f'<p class="top_words">{top_word_list[:-2]}</p>')
    grand_html.append(f'<p class="least_words">{least_word_list[:-2]}</p>')
    grand_html.append("</td>")
    grand_html.append("</tr>")

# save html to text file
ending = str(args.dataset_name) + "-table.html"
final_html_path = args.html_output_dir / ending

with open(final_html_path, "w", encoding="utf-8") as file:
    file.write("\n".join(grand_html))
print("HTML default table saved!")
