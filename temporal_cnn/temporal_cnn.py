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

import math
import os
import sys

from paddle.trainer_config_helpers import * 

############### Parameters ###############
node_num = 40
batch_size = 160
default_decay_rate(1e-4 * batch_size * node_num)
default_num_batches_regularization(1)
#default_initial_std(1e-4)
#default_initial_strategy(0)
default_momentum(0.9)

num_class = 4716

static = False
lr = 1

define_py_data_sources2(train_list='train.list',
                        test_list='test.list',
                        module='data_provider',
                        obj='processData')

############### Algorithm Configuration ###############
Settings(algorithm='sgd',
         batch_size=batch_size,
         learning_rate=1e-2 / (batch_size * node_num),
         learning_method='momentum',
         learning_rate_decay_a=0.1,
         learning_rate_decay_b= 4900000 * 10,
         learning_rate_schedule="discexp", )

############### Network Configuration ###############
def bn_relu_conv(name, input, filter_size, num_filters,
                  stride, padding, channels=None):

    tmp = batch_norm_layer(name=name + "_bn",
                         input=input,
                         act=ReluActivation())
    return img_conv_layer(name=name + "_conv",
                         input=tmp,
                         filter_size=1,
                         filter_size_y=filter_size,
                         num_channels=channels,
                         num_filters=num_filters,
                         stride_y=stride,
                         stride=1,
                         padding_y=padding,
                         padding=0,
                         act=LinearActivation(),
                         bias_attr=False)

def conv_bn_layer(name, input, filter_size, num_filters,
                  stride, padding, channels=None,
                  active_type=ReluActivation()):

    tmp = img_conv_layer(name=name + "_conv",
                         input=input,
                         filter_size_y=filter_size,
                         filter_size=1,
                         num_channels=channels,
                         num_filters=num_filters,
                         stride_y=stride,
                         stride=1,
                         padding=0,
                         padding_y=padding,
                         act=LinearActivation(),
                         bias_attr=False)
    return batch_norm_layer(name=name + "_bn",
                            input=tmp,
                            act=active_type)

def bn_relu_layer(name, input, active_type=ReluActivation()):
    return batch_norm_layer(name=name, 
                            input = input,
                            act=active_type)

def bottleneck_block(name, input, num_filters2, num_filters1):
    last_name = bn_relu_layer(name=name + "bn",
                              input = input)
    last_name = conv_bn_layer(name=name + '_branch2a',
                              input=last_name,
                              filter_size=3,
                              num_filters=num_filters2,
                              stride=1,
                              padding=1)
    
    last_name = img_conv_layer(name=name+"_branch2b",
                              input=last_name,
                              filter_size=1,
                              filter_size_y=3,
                              num_filters=num_filters1,
                              stride_y=1,
                              stride=1,
                              padding=0,
                              padding_y=1,
                              act=LinearActivation(),
                              bias_attr=False)

    return addto_layer(name=name + "_addto",
                       input=[input, last_name],
                       act=LinearActivation())

def deep_res_net():
    img = data_layer(name='input', size=1152*320, height=320, width=1)

    tmp = conv_bn_layer("conv1", img,
                        filter_size=7,
                        channels=1152,
                        num_filters=1024,
                        stride=2,
                        padding=3)

    tmp = img_pool_layer(name="pool1", input=tmp, pool_size=1, 
                         pool_size_y=3, stride=1, stride_y=2, padding=0, padding_y=0)

    tmp = bn_relu_conv("res2_0", tmp,
                        filter_size=1,
                        channels=1024,
                        num_filters=1024,
                        stride=2,
                        padding=0)

    for i in xrange(1, 4):
        tmp = bottleneck_block(name="res2_" + str(i),
                               input=tmp,
                               num_filters2=256, num_filters1=1024)

    tmp = img_pool_layer(name="pool2", input=tmp, pool_size=1, 
                        pool_size_y=3, stride=1, stride_y=2, padding=0, padding_y=0)

    tmp = bn_relu_conv("res3_0", tmp,
                        filter_size=1,
                        channels=1024,
                        num_filters=2048,
                        stride=1,
                        padding=0)

    for i in xrange(1, 4):
        tmp = bottleneck_block(name="res3_" + str(i),
                               input=tmp,
                               num_filters2=512, num_filters1=2048)


    tmp = img_pool_layer(name="pool3", input=tmp, pool_size=1, 
                        pool_size_y=3, stride=1, stride_y=2, padding=0, padding_y=0)

    tmp = bn_relu_conv("res4_0", tmp,
                        filter_size=1,
                        channels=2048,
                        num_filters=4096,
                        stride=1,
                        padding=0)

    for i in xrange(1, 4):
        tmp = bottleneck_block(name="res4_" + str(i),
                               input=tmp,
                               num_filters2=1024, num_filters1=4096)

    tmp = bn_relu_layer(name="last_bn",
                              input = tmp)

    tmp = img_pool_layer(name='avgpool',
                         input=tmp,
                         pool_size=1,
                         pool_size_y=10,
                         stride=1,
                         pool_type=AvgPooling())

    tmp = fc_layer(name='fc',
                      input=tmp,
                      size=4096,
                      layer_attr=ExtraLayerAttribute(drop_rate=0.5),
                      param_attr=ParamAttr(initial_std=0.01, learning_rate=lr),
                      bias_attr=ParamAttr(initial_std=0., l2_rate=0., learning_rate=lr),
                      act=LinearActivation())

    output = fc_layer(name='output_ft',
                      input=tmp,
                      size=num_class,
                      param_attr=ParamAttr(initial_std=0.01, learning_rate=lr),
                      bias_attr=ParamAttr(initial_std=0., l2_rate=0., learning_rate=lr),
                      act=SigmoidActivation())

    label=data_layer(name='label',size=num_class)

    tmp=multi_binary_label_cross_entropy(name='cost',input=output, label=label)

Inputs("input", "label")
Outputs("cost")
deep_res_net()

