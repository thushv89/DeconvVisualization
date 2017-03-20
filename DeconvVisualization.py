import tensorflow as tf
import numpy as np
import os
import getopt
import sys
import logging
from six.moves import cPickle as pickle
import load_data
from scipy.misc import imsave
from skimage.transform import rotate
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, some_other_arg):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')

__author__ = 'Thushan Ganegedara'

''' #################################################################################
#                                                                                   #
#              Visualizing and Understanding Convolutional Networks                 #
#                                                                                   #
#                      https://arxiv.org/pdf/1311.2901v3.pdf                        #
################################################################################# '''

''' #################################################################################
Notation used within code to explain what's happening
B - batch size, H - image height, W - image width,
D - depth/neuron count of a particular layer (conv or fully-connected)
L - number of layers, n - number of selected feature ids for a layer
################################################################################# '''

def get_max_activations(tf_data_batch,tf_filter_ids):
    '''
    This method calculate the maximally activated image indices
    and their corresponding activation value for a given set of filter IDs
    :param tf_data_batch: current batch of data
    :param tf_filter_ids: L x n matrix of filter IDs of to calculate the activation for
    where L is the number of layers (including pooling (ignored)) and
    N is the number of filter ids per layer

    :return: Two dictionaries with maximum values and indices
    '''
    global cnn_ops, last_2d_op, hyperparameters
    global weights, biases
    global logger

    max_outputs = {}
    max_indices = {}
    x = tf_data_batch
    prev_op = None

    # this needs to be n x L because we gather using innermost dimension
    tf_filter_ids = tf.transpose(tf_filter_ids)

    logger.info('Current set of operations: %s'%cnn_ops)

    logger.debug('Received data for X(%s)...'%x.get_shape().as_list())

    logger.info('Performing the specified operations ...')

    #need to calculate the output according to the layers we have
    for op_idx, op in enumerate(cnn_ops):
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,hyperparameters[op]['weights'],hyperparameters[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(x.get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            x = tf.nn.conv2d(x, weights[op], hyperparameters[op]['stride'], padding=hyperparameters[op]['padding'])
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(x.get_shape().as_list(),biases[op].get_shape().as_list()))
            x = tf.nn.relu(x + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,x.get_shape().as_list()))


            filter_ids_for_layer = tf.gather_nd(tf_filter_ids,[[op_idx]])
            x_mean = tf.reduce_mean(x,reduction_indices=[1,2])
            # x_mean => shape =[B x D]
            # tf.reduce_max(x_mean) => shape=[D]
            max_outputs[op] = tf.gather_nd(tf.reduce_max(x_mean,reduction_indices=[0]),
                                           tf.reshape(filter_ids_for_layer,shape=[-1,1]))

            max_outputs[op] = tf.gather_nd(tf.argmax(x_mean,dimension=0),
                                           tf.reshape(filter_ids_for_layer,shape=[-1,1]))

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,hyperparameters[op]['kernel'],hyperparameters[op]['stride']))
            x = tf.nn.max_pool(x,ksize=hyperparameters[op]['kernel'],strides=hyperparameters[op]['stride'],padding=hyperparameters[op]['padding'])
            logger.debug('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

        if 'fulcon' in op:

            # we need to reshape the output of last subsampling layer to
            # convert 4D output to a 2D input to the hidden layer
            # e.g subsample layer output [B x H x W x D] -> [B x H*W*D]
            if prev_op==last_2d_op:
                shape = x.get_shape().as_list()
                rows = shape[0]
                print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyperparameters[op]['in'])))
                x = tf.reshape(x, [rows,hyperparameters[op]['in']])

            if op != cnn_ops[-1]:
                x = tf.nn.relu(tf.matmul(x, weights[op]) + biases[op])
            else:
                x = tf.nn.softmax(tf.matmul(x, weights[op]) + biases[op])

            filter_ids_for_layer = tf.gather_nd(tf_filter_ids,[[op_idx]])
            # x => shape=[B x D]
            # reduce_max => shape=[D]
            # gather => [n]
            max_outputs[op] = tf.gather(tf.reduce_max(x,reduction_indices=[0]),
                                           tf.reshape(filter_ids_for_layer,shape=[-1,1]))

            max_outputs[op] = tf.gather(tf.argmax(x,dimension=0),
                                           tf.reshape(filter_ids_for_layer,shape=[-1,1]))

        prev_op = op

    return max_indices,max_outputs

# make one of this tf_op for each conv fc layers
def fwd_and_deconv_single_filter(layer_id, tf_feature_map_id,tf_max_data, guided=False):
    global weights,biases
    global cnn_ops, hyperparameters, last_2d_op
    global pool_switches
    global logger

    pool_switch_update_ops = []

    x = tf_max_data
    logger.info('Current set of operations: %s' % cnn_ops)
    logger.info('\tDefining Conv/Deconv for layer %s' %cnn_ops[layer_id])
    logger.debug('\tDefining data X of shape :%s...' % x.get_shape().as_list())

    x_layer = None
    for op in cnn_ops:
        if 'conv' in op:
            logger.debug(
                '\tConvolving (%s) With Weights:%s Stride:%s' % (op, hyperparameters[op]['weights'], hyperparameters[op]['stride']))
            logger.debug('\t\tX before convolution:%s' % (x.get_shape().as_list()))
            logger.debug('\t\tWeights: %s', weights[op].get_shape().as_list())
            x=tf.nn.conv2d(x, weights[op], hyperparameters[op]['stride'], padding=hyperparameters[op]['padding'])

            if guided:
                raise NotImplementedError

            logger.debug('\t\t Relu with x(%s) and b(%s)' % (
                x.get_shape().as_list(), biases[op].get_shape().as_list())
                         )
            x = tf.nn.relu(x + biases[op])

            if layer_id == cnn_ops.index(op):
                x_layer = tf.identity(x)

        if 'pool' in op:
            logger.debug(
                '\tPooling (%s) with Kernel:%s Stride:%s' % (op, hyperparameters[op]['kernel'], hyperparameters[op]['stride']))
            x, switch = tf.nn.max_pool_with_argmax(x, ksize=hyperparameters[op]['kernel'],
                                                          strides=hyperparameters[op]['stride'],
                                                          padding=hyperparameters[op]['padding'])

            pool_switch_update_ops.append(tf.assign(pool_switches[op],switch))
            logger.debug('\t\tX after %s:%s' % (op, x.get_shape().as_list()))

        if 'fulcon' in op:

            if prev_op==last_2d_op:
                shape = x.get_shape().as_list()
                rows = shape[0]
                print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyperparameters[op]['in'])))
                x = tf.reshape(x, [rows,hyperparameters[op]['in']])

            if guided:
                raise NotImplementedError

            if op != cnn_ops[-1]:
                x = tf.nn.relu(tf.matmul(x, weights[op]) + biases[op])
            else:
                x = tf.nn.softmax(tf.matmul(x, weights[op]) + biases[op])

        prev_op = op

    if 'conv' in cnn_ops[layer_id]:
        B, H, W, D = x_layer.get_shape().as_list()

        # Creating a tensor with the exact shape of the output of the layer
        # we're trying to visualize, that will have non-zeros only for the
        # position of the particular feature map we're trying to visualize
        values = tf.transpose(x_layer,[3,0,1,2])
        index = tf.reshape(tf_feature_map_id,shape=[-1,1])
        # expand_dims is essential to match the number of dimensions of
        # values and update
        updates = tf.expand_dims(tf.gather_nd(values,index),0)
        result = tf.scatter_nd(index,updates,tf.constant(values.get_shape().as_list()))
        result = tf.transpose(result,[1,2,3,0])

        assert x.get_shape().as_list() == result.get_shape().as_list()

        #

    elif 'fulcon' in cnn_ops[layer_id]:
        B, D = x_layer.get_shape().as_list()



pool_switches = None # dictionary of tf.variable
forward_mask = None # used for guided backpropagation

if __name__=='__main__':

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",["backprop_dir="])
    except getopt.GetoptError as err:
        print('<filename>.py --backprop_dir=')

    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--backprop_dir':
                backprop_feature_dir = arg

    logger = logging.getLogger('deconv_logger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('[%(funcName)s] %(message)s'))
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    graph = tf.Graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as session:

        tf_data_batch = tf.placeholder(shape=[batch_size,width,height,channels],dtype=tf.float32)
        tf_filter_ids = tf.placeholder(shape=[filters_per_layer],dtype=tf.int32)

        restore_weights_and_biases_and_hyperparameters(
            session,output_dir + os.sep + 'cnn-weights-31',output_dir + os.sep + 'cnn-biases-31')
        load_valid_dataset(dataset_type)

        # tf Operations

        # give the max indices and value of activations for all layers
        # 2 dictionaries with cnn_op => (N) Matrix
        tf_max_value,tf_max_index = get_max_activations(tf_data_batch,tf_filter_ids)

        for batch_id in range(dataset.shape[0] // batch_size):
