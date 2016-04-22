""" lambda function wrapper """

import gzip
import pickle
import logging
from timeit import default_timer as timer

CLASSES = {0: "negative", 4: "positive"}
LOGGER = logging.getLogger(__name__)
LOGGER.info("Loading files from FS")

MODEL_FILE = '../data/model.dat.gz'
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

    LOGGER.debug("Invoked lambda handler with event %s", event)

    # read data from event
    assert event, "AWS Lambda event parameter not provided"

    text = event.get("text")  # query text

    # input validation
    assert isinstance(text, (str))

    # call predicting function
    return predict(text)


def predict(text):
    """
        Predict the sentiment of a string
        @text: string - the string to be analyzed
    """

    LOGGER.info("extracting features...")
    start = timer()
    x_vector = MODEL.vectorizer.transform([text])
    end = timer()
    LOGGER.info("elapsed time: %s seconds", (end - start))

    LOGGER.info("predicting...")
    start = timer()
    y = MODEL.predict(x_vector)
    end = timer()
    LOGGER.info("elapsed time: %s seconds", (end - start))

    return [CLASSES.get(pred) for pred in y]
