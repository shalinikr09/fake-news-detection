import extras
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split
from sklearn import metrics

EMBEDDING_SIZE = 100
MAX_SEQ_LENGTH = 2000
VOCABULARY_SIZE = 20000
MAX_INPUT_SEQ_LENGTH = 2000

def gloveLSTM():
    np.random.seed(42)
    data_dir_path = './data'
    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/output4.csv")
    # Set `y`
    Y = [1 if label == 'REAL' else 0 for label in df.label]
    # Drop the `label` column
    df.drop("label", axis=1)

    df = df[df.text.apply(lambda x: x !="")]
    df['summary'] = df['summary'].map(lambda x: extras.clean_text(x))

    X = df['summary']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    tokenizer = Tokenizer(num_words= VOCABULARY_SIZE)

    tokenizer.fit_on_texts(Xtrain)
    sequencesTrain = tokenizer.texts_to_sequences(Xtrain)
    Xtrain = pad_sequences(sequencesTrain, maxlen=MAX_SEQ_LENGTH)

    tokenizer.fit_on_texts(Xtest)
    sequencesTest = tokenizer.texts_to_sequences(Xtest)
    Xtest = pad_sequences(sequencesTest, maxlen=MAX_SEQ_LENGTH)

    embeddings_index = dict()
    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((VOCABULARY_SIZE, 100))
    for word, index in tokenizer.word_index.items():
        if index > VOCABULARY_SIZE - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=MAX_INPUT_SEQ_LENGTH, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',extras.f1_m,extras.precision_m, extras.recall_m])

    print(model.summary())
    model.fit(Xtrain, Ytrain, epochs=1, batch_size=64)
    # Final evaluation of the model
    # scores = model.evaluate(Xtest, Ytest, verbose=1)
    loss, accuracy, f1, precision, recall = model.evaluate(Xtest, Ytest, verbose=1)

    # pred = model.predict(Xtest, verbose = 0)
    # pred = pred[:,0]

    print("accuracy:   %0.3f" % accuracy)
    print("f1:   %0.3f" % f1)
    print("recall:   %0.3f" % recall)
    print("precision:   %0.3f" % precision)

gloveLSTM()
