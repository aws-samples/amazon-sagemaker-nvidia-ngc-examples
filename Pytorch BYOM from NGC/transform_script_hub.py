
import torch
import os
from six import BytesIO
import numpy as np

def model_fn(model_dir):
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32',map_location='gpu')
    return model
                       
def input_fn(request_body, request_content_type):
    return torch.load(BytesIO(request_body))