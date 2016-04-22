import sys
from timeit import default_timer as timer
import gzip
import pickle

CLASSES = {0: "negative", 4: "positive"}


def load_model(model_filename):
	print "loading the model..."
	start = timer()
	try:
		with gzip.open(model_filename, 'rb') as f:
			model = pickle.load(f)
	except:
		raise IOError("can't find the trained model.")

	end = timer()
	print "elapsed time: %s seconds" % (end - start)

	return model

def predict(model, text):
    
    print "extracting features..."
    start = timer()
    x_vector = model.vectorizer.transform([text])
    end = timer()
    print "elapsed time: %s seconds" % (end - start)
    
    print "predicting..."
    start = timer()
    y = model.predict(x_vector)
    end = timer()
    print "elapsed time: %s seconds" % (end - start)

    return [CLASSES.get(pred) for pred in y]

def main(argv):
    """ Predict the sentiment of some text """

    text = argv[1]
    model_filename = "data/model.dat.gz"
    model = load_model(model_filename)
    print predict(model, text)


if __name__ == '__main__':
    main(sys.argv)