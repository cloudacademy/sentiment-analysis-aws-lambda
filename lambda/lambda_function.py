""" lambda function wrapper """

import gzip
import pickle

CLASSES = {0: "negative", 4: "positive"}

MODEL_FILE = 'model.dat.gz'
try:
    with gzip.open(MODEL_FILE, 'rb') as f:
        MODEL = pickle.load(f)
except:
    raise IOError("can't find the trained model at %s" % MODEL_FILE)

# pylint: disable=unused-argument
def lambda_handler(event, context=None):
    """
        Validate parameters and call the recommendation engine
        @event: API Gateway's POST body;
        @context: LambdaContext instance;
    """

    # input validation
    assert event, "AWS Lambda event parameter not provided"
    text = event.get("text")  # query text
    assert isinstance(text, str)

    # call predicting function
    return predict(text)


def predict(text):
    """
        Predict the sentiment of a string
        @text: string - the string to be analyzed
    """

    x_vector = MODEL.vectorizer.transform([text])
    y = MODEL.predict(x_vector)

    return CLASSES.get(y[0])
