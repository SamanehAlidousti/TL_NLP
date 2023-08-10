# Transfer Learning for NLP

This is a hands-on project on transfer learning for natural language processing with TensorFlow and TF Hub. The goal is to detect toxic contents.

## Objectives
- Use pre-trained NLP text embedding models from [TensorFlow Hub](https://tfhub.dev/) which is a repository of pre-trained TensorFlow models.

 

- Perform transfer learning to fine-tune models on real-world text data  

- Visualize model performance metrics with [TensorBoard](https://www.tensorflow.org/tensorboard)

## Requirements
I run this project on Google Colab.
Make sure to choose GPU as the Hardware accelerator. 


## Installation
Run the below cell on your colab notebook to install TensorFlow documentation package from GitHub repository.


```bash
  !pip install -q git+https://github.com/tensorflow/docs 
```
## Dataset
The dataset and its full description are available on [Quora Insincere Questions Classification data](https://www.kaggle.com/c/quora-insincere-questions-classification/data), but since I am using a pre-trained model, a subset of this would be enough which is available on [https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip](https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip).

## Overview
The text data consists of questions and corresponding labels. Consider a question vector as a distributed representation of a question that is computed for every question in the training set. The question vector along with the output label is then used to train the statistical classification model. 

The intuition is that the question vector captures the semantics of the question and, as a result, can be effectively used for classification. 

To obtain question vectors, there are two alternatives that have been used for several text classification problems in NLP: 
* word-based representations and 
* context-based representations
#### Word-based Representations

- A **word-based representation** of a question combines word embeddings of the content words in the question. We can use the average of the word embeddings of content words in the question. An average of word embeddings have been used for different NLP tasks.
- Examples of pre-trained embeddings include:
  - **Word2Vec**: These are pre-trained embeddings of words learned from a large text corpus. Word2Vec has been pre-trained on a corpus of news articles with  300 million tokens, resulting in 300-dimensional vectors.
  - **GloVe**: has been pre-trained on a corpus of tweets with 27 billion tokens, resulting in 200-dimensional vectors.
#### Context-based Representations

- **Context-based representations** may use language models to generate vectors of sentences. So, instead of learning vectors for individual words in the sentence, they compute a vector for sentences on the whole, by taking into account the order of words and the set of co-occurring words.
- Examples of deep contextualized vectors include:
  - **Embeddings from Language Models (ELMo)**: uses character-based word representations and bidirectional LSTMs. The pre-trained model computes a contextualized vector of 1024 dimensions. ELMo is available on Tensorflow Hub.
  - **Universal Sentence Encoder (USE)**: The encoder uses a Transformer architecture that uses attention mechanism to incorporate information about the order and the collection of words. The pre-trained model of USE that returns a vector of 512 dimensions is also available on Tensorflow Hub.
  - **Neural-Net Language Model (NNLM)**: The model simultaneously learns representations of words and probability functions for word sequences, allowing it to capture the semantics of a sentence. We will use pre-trained models available on Tensorflow Hub, that are trained on the English Google News 200B corpus, and computes a vector of 128 dimensions for the larger model and 50 dimensions for the smaller model.
Tensorflow Hub provides a number of [modules](https://tfhub.dev/s?module-type=text-embedding&tf-version=tf2&q=tf2) to convert sentences into embeddings such as Universal sentence ecoders, NNLM, BERT, and Wikiwords.

# Instruction
To use models without fine-tuning, set the attribute trainable=False, then fine-tuning is avoided, which means that the parameters of internal nodes will not be trained.
To have a better performance, by setting trainable=True, fine-tuning is done and the number of trainable parameters increases significantly.
