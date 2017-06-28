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
import gzip
import random
import cPickle
import numpy as np

from paddle.trainer.PyDataProvider2 import *

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """
    Dequantize the feature from the byte format to the float format
    """

    assert max_quantized_value >  min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    
    return feat_vector * scalar + bias

def on_init(settings, **kwargs):
    """
    Define the type and size of input feature and label
    """

    settings.input_types = [
        dense_vector(320 * 1152),
        sparse_non_value_slot(4716),
    ]

def arrangeData(lst, channel_size, max_length):
    """
    Padding zeros to make fixed length data
    """
    len_lst = len(lst)
    padding_left = (max_length - len_lst) / 2
    padding_right = max_length - len_lst - padding_left
    if padding_left < 0 or padding_right < 0:
        print "error dp padding"
        return None

    ret_data = []
    for j in range(channel_size):
        for i in range(max_length):
            if i < padding_left:
                now_data = 0.
            elif i >= padding_left + len_lst:
                now_data = 0.
            else:
                now_data = lst[i-padding_left][j]
            ret_data.append(now_data)
    return ret_data

@provider(use_seq=True,
        init_hook=on_init,
        should_shuffle=True, pool_size=1000)
def processData(obj, file_name):
    """
    Get data and yield to network
    Note: all data have been transformed to cPickle format for python
    """
                    
    with open(file_name, "r") as flist:
        for file_line in flist:
            inp = open(file_line.strip(), 'rb')
            data = cPickle.load(inp)
            inp.close()

            indexes = range(len(data))
            random.shuffle(indexes)

            for i in indexes:
                record = data[i]
                rgb   = record['feature']
                audio = record['audio']
                label = record['label']
                video = record['video']

                rgb = rgb[0:300,:]
                rgb_decode = Dequantize(rgb, 2, -2)
                rgb_decode = rgb_decode.astype(np.float32)

                audio = audio[0:300,:]
                audio_decode = Dequantize(audio, 2, -2)
                audio_decode = audio_decode.astype(np.float32)

                nframes = rgb_decode.shape[0]
                vec = []
                for row in range(nframes):
                    vec.append( rgb_decode[row,:].tolist() + audio_decode[row,:].tolist() )
                
                if len(vec) <= 320:
                    yield arrangeData(vec, 1152, 320), label
                else:
                    yield arrangeData(vec[-320:], 1152, 320), label

