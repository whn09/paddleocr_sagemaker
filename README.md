# paddleocr_sagemaker

Deploy PaddleOCR model on Amazon SageMaker.

## Build container

`./build_and_push.sh paddleocr`

# Deploy model to Amazon SageMaker Endpoint

`python create_endpoint -e {ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/paddleocr`

## Local test

```
nvidia-docker run -v -d -p 8080:8080 paddleocr
python local_predict-x-image.py
```

## Use AWS Lambda and API Gateway

```
cd lambda
./compile_and_upload.sh
```

Download python.zip and create a AWS Lambda Layer named `requests_toolbelt_sagemaker`, and create a AWS Lambda Function named `paddleocr` using `python3.6`.

Upload lambda_function_x-image.py to the Function, and add the Layer `requests_toolbelt_sagemaker` to the Function, and click Deploy, and attach `AmazonSageMakerFullAccess` to the Function's role.

Add API Gateway as a trigger, and then test the API using:

`curl -F "file=@./1.jpg; filename='1.jpg'" https://XXX.execute-api.us-east-1.amazonaws.com/default/paddleocr `