""" lambda function wrapper """

import gzip
import pickle

CLASSES = {
    0: "negative",
    4: "positive"
}

MODEL_FILE = 'model.dat.gz'
with gzip.open(MODEL_FILE, 'rb') as f:
    MODEL = pickle.load(f)

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
    assert isinstance(text, basestring)

    # call predicting function
    return predict(text)


def predict(text):
    """
        Predict the sentiment of a string
        @text: string - the string to be analyzed
    """

    x_vector = MODEL.vectorizer.transform([text])
    y_predicted = MODEL.predict(x_vector)

    return CLASSES.get(y_predicted[0])
