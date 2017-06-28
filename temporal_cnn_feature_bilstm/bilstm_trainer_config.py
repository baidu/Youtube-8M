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
from paddle.trainer_config_helpers import *

############### Parameters ###############
load_data_args = "ftr_dim:1024;label_size:4716"

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
settings(
    learning_rate=1e-03,
    batch_size=240,
    learning_method=RMSPropOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25,
)

############### Network Configuration ###############
feat_size = 1024
label_size = 4716
lstm_size = 1024
bias_attr = ParamAttr(initial_std=0.,l2_rate=0.)
layer_attr = ExtraAttr(drop_rate=0.5)

input = data_layer(name='feat', size=feat_size)
label = data_layer(name='label', size=label_size)

bi_lstm = bidirectional_lstm(input=input, size=lstm_size, return_seq=True)
dropout = dropout_layer(input=bi_lstm, dropout_rate=0.5)

lstm_weight = fc_layer(input=dropout, size=1, 
                       act=SequenceSoftmaxActivation(),
                       bias_attr=False)
scaled = scaling_layer(weight=lstm_weight, input=dropout)
lstm_pool = pooling_layer(input=scaled, pooling_type=SumPooling())

up_proj = fc_layer(input=lstm_pool, size=8192, 
                   act=ReluActivation(), 
                   bias_attr=bias_attr)

hidden = fc_layer(input=up_proj, size=4096, 
                  act=TanhActivation(), 
                  bias_attr=bias_attr)

output = fc_layer(input=hidden, size=label_size, name='output',
                  bias_attr=bias_attr,
                  act=SigmoidActivation())

cost = multi_binary_label_cross_entropy(name='cost', input=output, label=label)

outputs(cost)
