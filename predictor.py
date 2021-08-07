# -*- coding: utf-8 -*-
import io
import sys
import json
import boto3
import os
import warnings
import numpy as np
from paddleocr import PaddleOCR
import cv2
import time

from PIL import Image

warnings.filterwarnings("ignore",category=FutureWarning)


import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task

import flask

# The flask app for serving predictions
app = flask.Flask(__name__)


def execCmd(cmd):  
    r = os.popen(cmd)  
    text = r.read()  
    r.close()  
    return text  

print(execCmd('df -h'))

s3_client = boto3.client('s3')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def bbox_main(imgpath, detect='paddle'):
    # 主函数：输入图像路径返回key-val匹配后的字典，字典key为名称，val为的bbox的四个顶点
    # input：str，图像路径
    # output：dict，dict，保存key-val

    (filepath, tempfilename) = os.path.split(imgpath)
    (filename, extension) = os.path.splitext(tempfilename)

    # make sure the model parameters exist
    for i in ['/opt/program/inference/ch_ppocr_server_v2.0_det_infer',
              '/opt/program/inference/ch_ppocr_server_v2.0_rec_infer',
              '/opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer']:
        if os.path.exists(i):
            print("<<<<pretrained model exists for :", i)
        else:
            print("<<< make sure the model parameters exist for: ", i)
            break

    if detect == 'paddle':
        print ("start!!!!")
        ocr = PaddleOCR(det_model_dir='/opt/program/inference/ch_ppocr_server_v2.0_det_infer',
                        rec_model_dir='/opt/program/inference/ch_ppocr_server_v2.0_rec_infer',
                        cls_model_dir='/opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer',
                        use_pdserving=False)  # need to run only once to download and load model into memory
        print ("test!!!!")

        img = cv2.imread(imgpath)
        img_shape = img.shape
        print('img shape: ', img_shape)

        result = ocr.ocr(imgpath, rec=True)
        print (result)

        # save results
        res2 = {}

        label = []
        confidence = []
        bbox = []
        for i in result:
            label.append(i[1][0])
            confidence.append(i[1][1])
            bbox.append(i[0])

        res2['label'] = label
        res2['confidence'] = confidence
        res2['bbox'] = bbox

        print ('<<<< res2', res2)
        return res2, img_shape
    else:
        return

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)
    
    if flask.request.content_type == 'application/x-image':
        image_as_bytes = io.BytesIO(flask.request.data)
        img = Image.open(image_as_bytes)
        download_file_name = '/tmp/tmp.jpg'
        img.save(download_file_name)
        print ("<<<<download_file_name ", download_file_name)
    else:
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)

        bucket = data['bucket']
        image_uri = data['image_uri']

        download_file_name = '/tmp/'+image_uri.split('/')[-1]
        print ("<<<<download_file_name ", download_file_name)

        try:
            s3_client.download_file(bucket, image_uri, download_file_name)
        except:
            #local test
            download_file_name = './1.jpg'

        print('Download finished!')
    # inference and send result to RDS and SQS

    print('Start to inference:')

    # LOAD MODEL
    label = ''
    try:
        start = time.time()
        res, img_shape = bbox_main(download_file_name, detect='paddle')
        end = time.time()
        print ("Done inference! ", end-start)
        inference_result = {
            'label': res['label'],
            'confidences': res['confidence'],
            'bbox': res['bbox'],
            'shape': img_shape
        }
    except Exception as exception:
        print(exception)
        inference_result = {}
        
    _payload = json.dumps(inference_result,ensure_ascii=False,cls=MyEncoder)

    return flask.Response(response=_payload, status=200, mimetype='application/json')