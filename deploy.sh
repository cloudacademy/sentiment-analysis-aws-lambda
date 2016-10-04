#!/bin/bash

BUCKET="YOUR_BUCKET_NAME" # bucket name
FILENAME="deployment-package.zip" # upload key
TMP_FOLDER="/tmp/lambda-env-tmp/" # will be cleaned
OUTPUT_FOLDER="/tmp/lambda-env/" # will be cleaned


HERE=${BASH_SOURCE%/*} # relative path to this file's folder
LAMBDA_FOLDER="$HERE/lambda/" # relative path
PYTHON_ENV_FILE="$HERE/dependencies.zip" # relative path
OUTPUT_FILE="$OUTPUT_FOLDER/$FILENAME" 

# create target folders
mkdir $TMP_FOLDER
mkdir $OUTPUT_FOLDER

# unzip environment to temporary folder (quietly)
echo "Unzipping environment"
unzip -q $PYTHON_ENV_FILE -d $TMP_FOLDER

# copy lambda function stuff to temporary folder
echo "Copying aws_lambda files"
cp -r $LAMBDA_FOLDER/* $TMP_FOLDER

# move there
echo "Zipping everything together"
cd $TMP_FOLDER
# zip everything to output folder (recursively and quietly)
zip -r -q $OUTPUT_FILE ./*
# move back
cd -

# upload to S3
echo "Uploading to S3"
aws s3 cp --acl public-read $OUTPUT_FILE s3://$BUCKET/$FILENAME
echo "https://s3.amazonaws.com/$BUCKET/$FILENAME"

# clean everything
echo "Cleaning"
rm -rf $TMP_FOLDER
rm -rf $OUTPUT_FOLDER

echo "Done"

