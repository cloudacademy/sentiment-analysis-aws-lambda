""" Test the AWS Lambda function """
import sys
import lambda_function


def main(argv):
    
    try:
        event = {"text": argv[1]}
    except:
        raise ValueError("please provide a text string as argument")

    print lambda_function.lambda_handler(event, None)

if __name__ == '__main__':
    main(sys.argv)