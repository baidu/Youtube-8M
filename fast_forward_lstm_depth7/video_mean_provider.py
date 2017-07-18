import os
import sys
import logging
import cPickle
import random
import numpy as np
import time

from paddle.trainer.PyDataProvider2 import *

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
)
logger = logging.getLogger('paddle_data_provider')
logger.setLevel(logging.INFO)

def initHook(obj, **kwargs):
    obj.label_size = 4716
    obj.input_types = [
         dense_vector_sequence(1024),
         dense_vector_sequence(128),
         sparse_non_value_slot(obj.label_size),
    ]

@provider(init_hook=initHook)
def processData(obj, file_name, **kwargs):
    f = open(file_name, 'r')
    all_lines = f.readlines()
    f.close()

    random.shuffle(all_lines)
    for line in all_lines:
        inp = open(line.strip(), 'rb')
        data = cPickle.load(inp)
        inp.close()

        indexes = range(len(data))
        random.shuffle(indexes)

        for i in indexes:
            record = data[i]
            rgb   = record['feature'].astype(np.float32)
            audio = record['audio'].astype(np.float32)
            label = record['label']
            video = record['video']
            nframes = rgb.shape[0]

            feat_list = [None for row in range(nframes)]
            audio_list = [None for row in range(nframes)]
            
            # data dropout 0.3
            length = len(feat_list)
            if length < 20:
                for row in range(nframes):
                    feat_list[row]  = rgb[row,:]
                    audio_list[row] = audio[row,:]
            else:
                feat_list  = feat_list[:length/3]
                audio_list = audio_list[:length/3]
                row_smp = 0
                select = int(random.random()*3)
                for row in range(nframes):
                    if row%3 == select:
                        feat_list[row_smp] = rgb[row,:]
                        audio_list[row_smp] = audio[row,:]
                        row_smp += 1
                    if row_smp == len(feat_list):
                        break
                if row_smp<30 or row_smp!=len(feat_list):
                    continue

            yield feat_list, audio_list, label

