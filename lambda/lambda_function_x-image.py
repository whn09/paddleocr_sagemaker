import boto3
from botocore.config import Config
from boto3.session import Session
import json
import io
import base64
from requests_toolbelt.multipart import decoder

config = Config(
    read_timeout=120,
    retries={
        'max_attempts': 0
    }
)

def infer(input_image):
    payload = input_image
    print('payload:', payload)

    sagemaker_runtime_client = boto3.client('sagemaker-runtime', config=config)
    session = Session(sagemaker_runtime_client)

#     runtime = session.client("runtime.sagemaker",config=config)
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName='paddleocr',
        ContentType="application/x-image",
        Body=payload)

    result = json.loads(response["Body"].read())
    print (result)
    return result

def lambda_handler(event, context):
    # TODO implement
    # print('event:', event)
    
    body = event["body"]
    content_type = event["headers"]["content-type"]
    body_dec = base64.b64decode(body)
    multipart_data = decoder.MultipartDecoder(body_dec, content_type)
    binary_content = []
    for part in multipart_data.parts:
        binary_content.append(part.content)

    image = io.BytesIO(binary_content[0])
    
    result = infer(image)
    return {
        'statusCode': 200,
        'body': json.dumps(result, ensure_ascii=False)
    }

