import os
import pickle
import gzip
import numpy as np
from sklearn import svm, naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from timeit import default_timer as timer



def main():
    """ Train and test a model """

    col_names = ["polarity", "id", "date", "query", "user", "text"]

    data_dir = "data"
    train_data_file = 'data/trainingandtestdata/training.1600000.processed.noemoticon.csv'
    test_data_file = 'data/trainingandtestdata/testdata.manual.2009.06.14.csv'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not all([os.path.isfile(file) for file in [train_data_file, test_data_file]]):
        print "dataset not found, downloading it..."
        download_dataset()
        print "ok"

    model_filename = "data/model.dat.gz"
    
    if not os.path.isfile(model_filename):
        print "model not found, training it..."
        train_dataset = pd.read_csv(
            train_data_file,
            names = col_names
        )
        model = train_model(train_dataset)
        print "saving the model..."
        pickle.dump(model, gzip.open(model_filename, "wb"))
    else:
        print "loading the model..."
        model = pickle.load(gzip.open(model_filename, "rb"))

    test_dataset = pd.read_csv(
        test_data_file,
        names = col_names
    )

    test_dataset = test_dataset[test_dataset.polarity != 2]
    
    test_model(model, test_dataset)

def train_vectorizer(corpus, max_features=10000):
    """ Train the vectorizer """
    print "training the vectorizer..."
    vectorizer = CountVectorizer(decode_error='ignore', max_features=max_features)
    vectorizer.fit(corpus)
    print "ok"
    return vectorizer

def extract_features(vectorizer, text):
    """ Extract text features """
    return vectorizer.transform(text)

def train_model(dataset):
    """ Train a new model """
    text_train = dataset.text
    vectorizer = train_vectorizer(text_train)
    vectorizer.stop_words_ = set({})
    print "extracting features..."
    X_train = extract_features(vectorizer, text_train)
    y_train = dataset.polarity
    # model = svm.LinearSVC()
    model = naive_bayes.MultinomialNB()
    print "training the model..."
    start = timer()
    model.fit(X_train, y_train)
    model.vectorizer = vectorizer
    print "ok"
    end = timer()
    print "elapsed time: %s seconds" % (end - start)
    return model

def test_model(model, dataset):
    """ Test the given model (confusion matrix) """
    print "testing the model..."
    text_test = dataset.text
    X_test = extract_features(model.vectorizer, text_test)
    y_test = dataset.polarity
    y_ = model.predict(X_test)
    C = confusion_matrix(y_test, y_)
    print np.around(C / C.astype(np.float).sum(axis=1) / 0.01)
    print "accuracy: %.3f" % (float(np.trace(C))/float(np.sum(C)))

def download_dataset():
    
    import urllib2
    import zipfile

    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    file_name = 'data/trainingandtestdata.zip'
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()

    with zipfile.ZipFile(file_name, "r") as z:
        folder = 'data/trainingandtestdata/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        z.extractall(folder)

if __name__ == '__main__':
    main()