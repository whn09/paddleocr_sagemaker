import io
import sys
import json
import os
from io import BytesIO
import numpy as np
from paddleocr import PaddleOCR
import time


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


def model_fn(model_dir):
    ocr = PaddleOCR(det_model_dir=os.path.join(model_dir, 'det_infer'),
                    rec_model_dir=os.path.join(model_dir, 'rec_infer'),
                    cls_model_dir=os.path.join(model_dir, 'cls_infer'),
                    use_pdserving=False)
    return ocr


def input_fn(request_body, request_content_type):
#     print('[DEBUG] request_body:', type(request_body))
#     print('[DEBUG] request_content_type:', request_content_type)
    
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    elif request_content_type == 'application/x-npy':
        np_bytes = BytesIO(request_body)
        return np.load(np_bytes, allow_pickle=True)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body
    
def predict_fn(input_data, model):
    result = model.ocr(input_data, rec=True)
    
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
    
    result_json = json.dumps(res2, ensure_ascii=False, cls=MyEncoder)
    
    return result_json


# def output_fn(prediction, content_type):
#     pass


if __name__ == "__main__":
    import cv2
    filename = '../../1.jpg'
    img = cv2.imread(filename)
    model = model_fn('../')
    result = predict_fn(img, model)
    print(result)
