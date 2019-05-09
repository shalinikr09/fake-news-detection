from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_fake_news_detector.library.utility.plot_utils import plot_and_save_history
from keras_fake_news_detector.library.classifiers.feedforward_networks import Doc2VecFeedforwardNet
from keras_fake_news_detector.library.encoders.doc2vec import DOC2VEC_MAX_SEQ_LENGTH
from keras_fake_news_detector.library.utility.news_loader import fit_input_text
import numpy as np
from sklearn import metrics



def main():
    np.random.seed(42)
    data_dir_path = './data'
    very_large_data_dir_path = './very_large_data'
    report_dir_path = './reports'

    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/output4.csv")

    print('extract configuration from input texts ...')
    Y = [1 if label == 'REAL' else 0 for label in df.label]
    X = df['summary']

    config = fit_input_text(X, max_input_seq_length=DOC2VEC_MAX_SEQ_LENGTH)
    config['num_target_tokens'] = 2

    print('configuration extracted from input texts ...')

    classifier = Doc2VecFeedforwardNet(config)
    classifier.load_glove(very_large_data_dir_path)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = classifier.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=100)

    history_plot_file_path = report_dir_path + '/' + Doc2VecFeedforwardNet.model_name + '-history.png'
    plot_and_save_history(history, classifier.model_name, history_plot_file_path)

    print('start predicting ...')
    pred = classifier.predict(Xtest)
    print(pred)
    accuracy = metrics.accuracy_score(Ytest, pred)
    f1 = metrics.f1_score(Ytest, pred)
    precision = metrics.precision_score(Ytest, pred)
    recall = metrics.recall_score(Ytest, pred)

    print("accuracy:   %0.3f" % accuracy)
    print("f1:   %0.3f" % f1)
    print("recall:   %0.3f" % recall)
    print("precision:   %0.3f" % precision)


if __name__ == '__main__':
    main()
