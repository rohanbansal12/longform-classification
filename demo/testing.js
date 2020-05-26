const fs = require('fs');
const mathjs = require('mathjs')

var rawdata = fs.readFileSync('../../data/demo-data/select_demo_articles.json');
var articles = JSON.parse(rawdata);
var rawdata = fs.readFileSync('../../data/demo-data/pub_emb+bias.json');
var publications = JSON.parse(rawdata)
var rawdata = fs.readFileSync('../../data/demo-data/word_to_emb+bias_dict.json')
var words = JSON.parse(rawdata)
var list_of_logits = []

articles.forEach(function(item) {
    var word_emb_product_sum = 0
    var word_count = 0
    var word_list = []
    var word_logit_list = []
    item.text.forEach(function(word) {
        if (words[word] != null && !(word_list.includes(word))) {
            word_count += 1
            var word_emb_product = mathjs.dot(publications.embedding, words[word].embedding) + words[word].bias;
            word_emb_product_sum += word_emb_product
            word_list.push(word)
            word_logit_list.push(word_emb_product)
        }
    })
    indexedWordLogits = word_logit_list.map(function(e,i){return {ind: i, val: e}})
    indexedWordLogits.sort(function(x, y){return x.val < y.val ? 1 : x.val == y.val ? 0 : -1});
    indices = indexedWordLogits.map(function(e){return e.ind});
    var top_words = []
    for (var idx = 0; idx < 10; idx++) {
        current_idx = indices[idx];
        current_word = word_list[current_idx];
        current_logit = word_logit_list[current_idx];
        top_words.push({word: current_word, contribution: current_logit});
    }
    var least_words = []
    for (var idx = indices.length-1; idx > indices.length-11; idx--) {
        current_idx = indices[idx];
        current_word = word_list[current_idx];
        current_logit = word_logit_list[current_idx];
        least_words.push({word: current_word, contribution: current_logit});
    }
    var word_emb_product_mean = mathjs.divide(word_emb_product_sum, word_count);
    var article_logit = word_emb_product_mean + publications.bias;
    item.logit = article_logit;
    item.top_words = top_words;
    item.least_words = least_words;
    list_of_logits.push(article_logit);
})

console.log("Logits Generated!")
indexedLogits = list_of_logits.map(function(e,i){return {ind: i, val: e}});
indexedLogits.sort(function(x, y){return x.val < y.val ? 1 : x.val == y.val ? 0 : -1});
indices = indexedLogits.map(function(e){return e.ind});
for (var top_idx = 0; top_idx < 50; top_idx++) {
    article = articles[indices[top_idx]];
    var current_html = ['<li class="entry">',
                        '<h4 class="article-title" href="' + article.url + '">' + article.title + '</h4>',
                        '<h5 class="logit">' + article.logit + '</h5>']
    article.top_words.forEach(function(entry){
        current_html.push('<p class="top-word">' + entry.word + ': ' + entry.contribution + '</p>')
    })
    article.least_words.forEach(function(entry){
        current_html.push('<p class="least-word">' + entry.word + ': ' + entry.contribution + '</p>')
    })
    current_html.push('</li>')
    current_html = current_html.join("\n")
    console.log(current_html)
}