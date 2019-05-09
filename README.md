# keras-fake-news-detection
Implementation of text summarization using RNN LSTM

### Word Embedding
Used [Glove pre-trained vectors](https://nlp.stanford.edu/projects/glove/), Word2Vec and Doc2Vec to initialize word embedding.

## Layers
Used LSTM cell layer along with a dense layer

## Requirements
- Python 3
- Keras (=2.2.4)


## Usage
### Prepare data
Dataset is available at [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet/). 

### Train

For training and testing the model using the three different embeddings:

$ python word2vecLSTM.py

$ python gloveLSTM.py

$ python doc2vecLSTM.py