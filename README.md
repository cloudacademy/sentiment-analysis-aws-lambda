# Sentiment analysis in the Cloud with AWS Lambda.

The Cloud Academy team shows how to build a sentiment analysis machine learning model by using a pubic dataset and how to deploy it to production with AWS Lambda and API Gateway.

## How to install requirements

Python requirements can be installed with pip.

    $ pip install -r requirements.txt
    
## Training phase

You can train and persist the model into file by executing:

    $ python main.py

## Test the model locally

You can run the model via CLI as follows:

    $ python predict.py "This function is awesome"
    > positive
    
## How to deploy the Lambda function

You have to create a new deployment package containing every Python dependency and the OS libraries required by scikit-learn and numpy.

[Here](https://github.com/ryansb/sklearn-build-lambda) you can find the whole stack ready to be uploaded  ([reference](https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/)).

Then, you can bind the AWS Lambda function to a new Amazon API Gateway resource and obtain a RESTful interface for your model.

### References

* [Cloud Academy & AWS: how we use Amazon Web Services for machine learning and data collection](https://www.slideshare.net/AlexCasalboni/cloud-academy-aws-how-we-use-amazon-web-services-for-machine-learning-and-data-collection/) (webinar)

* [Using Scikit-Learn In AWS Lambda](https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/) (article by [Ryan Brown](https://github.com/ryansb))

* [Sentiment Analysis dataset](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) (by cs.stanford.edu)

* [Amazon Serverless Architecture](http://cloudacademy.com/blog/amazon-serverless-api-gateway-lambda-cloudfront-s3/) (Cloud Academy blog)

* [Google Cloud Functions VS AWS Lambda](http://cloudacademy.com/blog/google-cloud-functions-serverless/) (Cloud Academy blog)