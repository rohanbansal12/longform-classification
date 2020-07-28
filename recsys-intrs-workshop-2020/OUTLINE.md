# intro

* recommendation of items described by sets is a pervasive problem: discuss
  examples, ICD9 codes. do not discuss document example.

* here is why building a recommendation system for ranking sets is difficult: it
  is a statistical challenge because the large number of items prohibits unique
  parameters for every item. this necessitates sharing information across items
  with similar sets of attributes. (figure 1: histogram of set sizes in meals
  illustrates the issue nicely).

* the set data structure is order invariant. give example of meals and the bag
  of words assumption of meal names (order of foods does not matter)

* talk about order-invariant models, how they include matrix factorization, etc.

* how do we choose amongst order-invariant models? we can think about optimizing
  for the evaluation metric we use: recall. this is a binary task, and
  classifiers are a good choice.

* we develop rankfromsets, a deep classifier. we show it can approximate any
  order-invariant algorithm with the neural network and residual architectures.
  discuss how we train it to descriminate likely sets from unlikely sets. we
  show it does well in experiments.

* related work.


# section: notation

* define notation for sets, users consuming sets.


# section: method - model

* lay out desiderata for recommendation model function:

  - output of the model can be used to rank items based on their sets of
  attributes

  - input is a user and an (order-invariant) set
  
  - scalable; can be fit to large datasets with tens of millions of unique sets
    (shares info across sets)

  - the function is flexible (can represent a general class of such functions)

* we build rankfromsets to satisfy the desiderata:

  - the model is a classifier that uses user embeddings and attribute embeddings
    to assign a probability of item consumption. these probabilities are used to
    rank items for recommendation.
  
  - summation of attribute embeddings preserves order-invariance
  
  - the absence of set embeddings makes it scalable (attribute embeddings are
    shared across sets).
  
  - flexibility: the residual and neural network models can approximate any
    function

* the model uses attribute attribute embeddings to share information across
  sets. this solves the cold-start problem if consumption data is not available.

* for items where consumption data is available, paragraph addressing how we do
  collaborative filtering (the above points only talk about using content):

  - the function $h$ is for collaborative filtering.
  
  - if the number of sets presents a statistical challenge, the model uses
    additional information about the item such as metadata, for intercepts. this
    allows collaborative filtering (for example, information about foods in
    meals)
  
  - if the number of sets is not a statistical challenge, uses item intercepts
    for collaborative filtering

* describe the choices for f in the model: inner product, neural network,
  residual architectures.

* foreshadowing: we prove that this model is a good choice next, by proving that
  recall can be viewed as a binary task (hence classifiers do well). we also
  prove rankfromsets can approximate any function, including the CTPF posterior
  predictive.


# section: theory - why the model is justified

* we evaluate recommendation models with recall. define it mathematically.

* theorem 1: recall is maximized by a classifier. this justifies rankfromsets.

* rankfromsets is also justified by its flexibility; by choosing a powerful
  function approximator such as a deep neural network, it can approximate many
  models that may not have the theoretical guarantee of directly optimizing an
  an evaluation metric.

* define order-invariant model for ranking from sets: such a model's output must
  be independent the order the set elements are fed to the model. assert that
  CTPF and permutation-marginalied neural networks are in this class (proved in
  appendix).

* theorem 2: the residual and neural network choices in rankfromsets can also
  approximate the functional form of any order-invariant model. note that this
  is true for the functional form, but the objective for training the model can
  change (e.g. if CTPF is stopped with held-out log-likelihood or recall; how do
  we train rankfromsets to approximate it?).


# empirical study

* describe meal data

* define sampled recall (for when the number of items/sets is large).

* describe hyperparameters word embedding model, recurrent neural network,
  rankfromsets. say that ctpf code failed to scale despite multiple attempts to
  run it.

* discuss the sampled recall plot, with the residual model doing best. mention
  qualitative results in table of nearest neighbors.

* describe arxiv data

* describe hyperparameters for CTPF, word embedding, recurrent neural network,
  rankfromsets.

* discuss the in/out-matrix recall & precision plot. interesting that word
  embedding model works for recall, not for precision. describe why LSTM failed.

* mention qualitative results on t-SNE plot.


# discussion

* future work: what if we get a new attribute? use n-gram like fastText to infer
  the attribute embedding.

* can we directly optimize a ranking loss like NDCG? this is unclear but should
  be possible.


# appendix

* show that matrix factorization (CTPF) and permutation-marginalized recurrent
  neural networks are part of the class.
