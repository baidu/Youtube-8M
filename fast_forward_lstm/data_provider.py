# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
import cPickle
import random
import numpy as np

from paddle.trainer.PyDataProviderWrapper import IndexSlot, provider, DenseSlot, SparseNonValueSlot

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
)
logger = logging.getLogger('paddle_data_provider')
logger.setLevel(logging.INFO)

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """
    Dequantize the feature from the byte format to the float format
    """

    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    
    return feat_vector * scalar + bias

def initHook(obj, *file_list, **kwargs):
    """
    Define the type and size of input feature and label
    """
    
    config_map = {}
    configs = [args.split(':') for args in kwargs['load_data_args'].split(';')]
    for confs in configs:
        config_map[confs[0]] = confs[1]
    
    logger.info(config_map)
    
    obj.ftr_dim = int(config_map.get('ftr_dim'))
    obj.label_size = int(config_map.get('label_size'))
    obj.slots = [DenseSlot(obj.ftr_dim), SparseNonValueSlot(obj.label_size)]

@provider(use_seq=True,init_hook=initHook)
def processData(obj, file_name):
    """
    Get data and yield to network
    Note: all data have been transformed to cPickle format for python
    """
    
    inp = open(file_name, 'rb')
    data = cPickle.load(inp)
    inp.close()

    indexes = range(len(data))
    random.shuffle(indexes)
    
    for i in indexes:
        record = data[i]
        nframes = record['nframes']
        rgb   = record['feature']
        audio = record['audio']
        label = record['label']
        video = record['video']
			
        rgb = rgb[0: nframes, :]
        rgb_decode = Dequantize(rgb, 2, -2)
        rgb_decode = rgb_decode.astype(np.float32)

        audio = audio[0: nframes, :]
        audio_decode = Dequantize(audio, 2, -2)
        audio_decode = audio_decode.astype(np.float32)

        feat_list = []
        for row in range(rgb_decode.shape[0]):
            feat_com = np.hstack((rgb_decode[row, :], audio_decode[row, :]))
            feat_list.append(feat_com)

        yield (feat_list, [label])

