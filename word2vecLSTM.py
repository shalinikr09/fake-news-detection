import extras
import numpy as np
import pandas as pd
import string
import gensim
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn import metrics

EMBEDDING_SIZE = 100
MAX_SEQ_LENGTH = 2000
VOCABULARY_SIZE = 20000
MAX_INPUT_SEQ_LENGTH = 2000

def word2vecLSTM():
    newsLines = list()
    data_dir_path = './data'
    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/output4.csv")

    lines = df['summary'].values.tolist()

    df = pd.read_csv(data_dir_path + "/output4.csv")
    # Set `y`
    Y = [1 if label == 'REAL' else 0 for label in df.label]
    # Drop the `label` column
    df.drop("label", axis=1)

    df = df[df.summary.apply(lambda x: x !="")]
    df['summary'] = df['summary'].map(lambda x: extras.clean_text(x))

    X = df['summary']

    for line in lines:
        tokens = word_tokenize(line)

        #conver to lower case
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        words = [word for word in stripped if word.isalpha()]

        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words] 
        newsLines.append(words)

    model = gensim.models.Word2Vec(sentences = newsLines, size = EMBEDDING_SIZE,  window = 5, workers = 4, min_count = 1)

    words = list(model.wv.vocab)
    print('Vocabulary size: %d' % len(words))

    filename = 'news_word2vec_politifact_summary.txt'
    model.wv.save_word2vec_format(filename, binary = False)

    embeddingsIndex = {}

    f = open(os.path.join('', 'news_word2vec_politifact_summary.txt'), encoding = 'utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddingsIndex[word] = coefs

    f.close()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    tokenizer = Tokenizer(num_words= VOCABULARY_SIZE)

    tokenizer.fit_on_texts(Xtrain)
    sequencesTrain = tokenizer.texts_to_sequences(Xtrain)
    Xtrain = pad_sequences(sequencesTrain, maxlen=MAX_SEQ_LENGTH)

    tokenizer.fit_on_texts(Xtest)
    sequencesTest = tokenizer.texts_to_sequences(Xtest)
    Xtest = pad_sequences(sequencesTest, maxlen=MAX_SEQ_LENGTH)


    embedding_matrix = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))

    for word, i in tokenizer.word_index.items():
        if i > VOCABULARY_SIZE - 1:
            continue
        embedding_vector = embeddingsIndex.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=MAX_INPUT_SEQ_LENGTH, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',extras.f1_m,extras.precision_m, extras.recall_m])

    print(model.summary())
    model.fit(Xtrain, Ytrain, epochs=1, batch_size=64)
    loss, accuracy, f1, precision, recall = model.evaluate(Xtest, Ytest, verbose=1)

    print("accuracy:   %0.3f" % accuracy)
    print("f1:   %0.3f" % f1)
    print("recall:   %0.3f" % recall)
    print("precision:   %0.3f" % precision)

word2vecLSTM()
