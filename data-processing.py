
# building vocab manually for text/title and unique articles

counter = collections.Counter()
url_counter = collections.Counter()
urls = []

for example in train.examples + val.examples + test.examples:
    counter.update(example.text)
    counter.update(example.title)
    urls.append(example.url)

url_counter.update(urls)
word_to_id = {word: id for id, word in enumerate(counter.keys())}
article_to_id = {word: id for id, word in enumerate(url_counter.keys())}


#Create a weighted sampler function to ensure that num of pos_samples = num of neg_samples

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
    
#Define collate_fn function to return the inputs needed for the forward function within the model

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
    return articles, word_attributes, attribute_offsets
