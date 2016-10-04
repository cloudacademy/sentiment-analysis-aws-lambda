""" Example of prediction from CLI """
import sys
import gzip
import pickle

CLASSES = {
    0: "negative",
    4: "positive"
}


def load_model(model_filename):
    """ Load model from file """
    print "loading the model..."
    try:
        with gzip.open(model_filename, 'rb') as fmodel:
            model = pickle.load(fmodel)
    except Exception as ex:
        raise IOError("Couldn't load  model: %r" % ex)

    return model

def predict(model, text):
    """ Predict class given model and input (text) """
    print "Extracting features..."
    x_vector = model.vectorizer.transform([text])
    y_predicted = model.predict(x_vector)
    return CLASSES.get(y_predicted[0])

def main(argv):
    """ Predict the sentiment of the given text """
    text = argv[1]
    model_filename = "data/model.dat.gz"
    model = load_model(model_filename)
    print predict(model, text)


if __name__ == '__main__':
    main(sys.argv)
