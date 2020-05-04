
import torch
import os
from six import BytesIO
import numpy as np

def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, 'nvidia_ssdpyt_fp32_190826.pt')
    return model
                       
def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/python-pickle':
        return torch.load(BytesIO(request_body))
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass