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
import math
from paddle.trainer_config_helpers import *

############### Parameters ###############
lr_hid_col = 4e-2

model_type('recurrent_nn')

feat_size = 1152
label_size = 4716
load_data_args = "ftr_dim:%d;label_size:%d"%(feat_size, label_size)

###
TrainData(PyData(
	files="train.list",
	load_data_module="data_provider",
	load_data_object = "processData",
	load_data_args = load_data_args,
	async_load_data = True)
)

TestData(PyData(
	files="test.list",
	load_data_module="data_provider",
	load_data_object = "processData",
	load_data_args = load_data_args)
)

############### Algorithm Configuration ###############
Settings(
    algorithm='sgd',
    learning_method='adadelta',
    learning_rate=1e-1,
    learning_rate_decay_a=0,
    learning_rate_decay_b=0,
    ada_rou=0.95,
    ada_epsilon=1e-6,
    batch_size=256,
)

lstm_size = 1024
lstm_depth = 2
with_attention = True

default_decay_rate(1e-4)
default_num_batches_regularization(1)
default_initial_std(1 / math.sqrt(lstm_size) / 3.0)

############### Network Configuration ###############
Inputs("feat", 'label')

Layer(name='feat', type='data', size=feat_size)
Layer(name='label', type='data', size=label_size)

def BidirectionStackLSTMmemory(name, input, size, depth, with_attention):
    global lr_hid_col
    for i in range(1, depth + 1):
        for direction in ["forward", "backward"]:
            # linear transform
            Layer(
                name = "{n}_{d}_input_recurrent{idx:02d}".format(n = name,d = direction,idx = i),
                type = "mixed",
                size = size * 4,
                bias = True,
                inputs = [FullMatrixProjection("{n}_{d}_input_recurrent{idx:02d}".format(n = name,d = direction,idx = i - 1),learning_rate=lr_hid_col),
                          FullMatrixProjection("{n}_{d}_lstm{idx:02d}".format(n = name,d = direction,idx = i - 1),initial_std=0)] if i > 1 else \
                         [FullMatrixProjection(input)]
            )

            # LSTM
            Layer(
                name="{n}_{d}_lstm{idx:02d}".format(n=name, d=direction, idx=i),
                type="lstmemory",
                reversed=not i % 2 if direction == 'forward' else i % 2,
                active_type="tanh",
                active_gate_type="sigmoid",
                active_state_type="tanh",
                bias=Bias(initial_std=0),
                inputs=Input(
                      "{n}_{d}_input_recurrent{idx:02d}".format(n=name, d=direction, idx=i),
                      initial_std=0.01), 
            )

    Layer(
        name = name,
        type = "concat",
        inputs = [name + "_forward_lstm{idx:02d}".format(idx = depth),\
                  name + "_backward_lstm{idx:02d}".format(idx = depth)] if with_attention else
                 [name + "_forward_last", name + "_backward_first"]
    )

###
BidirectionStackLSTMmemory(
    name="bidirection_stack_lstm",
    input="feat",
    size=lstm_size,
    depth=lstm_depth,
    with_attention=with_attention
)

Layer(
    name = "weights",
    inputs = "bidirection_stack_lstm",
    type = "fc",
    size = 1,
    active_type = "sequence_softmax",
    bias = False,
)

Layer(
    name = "scaling",
    inputs = ["weights", "bidirection_stack_lstm"],
    type = "scaling",
)

Layer(
    name = "sum_pooling",
    inputs = [Input("scaling")],
    type = "average",
    bias = False,
    active_type = "",
    average_strategy = 'sum'
)

Layer(
    name = "fc1",
    inputs = "sum_pooling",
    type = "fc",
    size = 8192,
    active_type = "relu",
    bias = Bias(initial_std=0)
)
    
Layer(
    name = "fc2",
    inputs = "fc1",
    type = "fc",
    size = 4096,
    active_type = "tanh",
    bias = Bias(initial_std=0)
)

Layer(
    name = "output",
    inputs = "fc2",
    type = "fc",
    size = label_size,
    active_type = "sigmoid",
    bias = Bias(initial_std=0)
)

### cost
Layer(
    type = 'multi_binary_label_cross_entropy',
    name = 'cost',
    inputs = [Input('output'), Input('label')]
)

Outputs('cost')
