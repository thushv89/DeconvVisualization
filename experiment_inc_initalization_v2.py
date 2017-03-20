__author__ = 'Thushan Ganegedara'

'''==================================================================
#   This is an experiment to check if incrementally adding layers   #
#   help to reach higher accuracy quicker than starting with all    #
#   the layers at once.
#   Version 2 uses a 1x1 pooling layer so you don't need both
#   conv_balance and pool_balance. pool_balance is enough.
#   and you directly change the weights to fully connected layer
=================================================================='''

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from math import ceil,floor
import load_data
import logging
import sys
import time
import qlearner
import learn_best_actions
from data_pool import Pool
import getopt
from scipy.misc import imsave
from skimage.transform import rotate
import load_data

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

logger = None
logging_level = logging.DEBUG
logging_format = '[%(funcName)s] %(message)s'

backprop_feature_dir = 'backprop_features'

#type of data training
dataset_type = 'imagenet-100'

if dataset_type=='cifar-10':
    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb
    train_size = 40000//2
    valid_size = 10000
    test_size = 10000

elif dataset_type=='imagenet-100':
    image_size = 224
    num_labels = 100
    num_channels = 3
    train_size = 128099
    valid_size = 5000

batch_size = 128 # number of datapoints in a single batch

incrementally_add_layers = False
save_weights_periodic = True if dataset_type=='imagenet-100' else False
save_period  = 5

# for cifar-10 start_lr 0.01/0.001 didn't work when starting with full network
start_lr = 0.1 if incrementally_add_layers else 0.01
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = False

include_l2_loss = False
# keep beta small (0.2 is too much >0.002 seems to be fine)
beta = 1e-1

# we use early stoppling like mechanisim to decay the learning rate
# if the validation accuracy plateus then we decay the learning rate by 0.9
# if we don't see an improvement of the validation accuracy after no_improvement_cap epochs
# we terminate the loop

validation_frequency = 1
valid_drop_cap = 2
no_improvement_cap = 10
incrementing_frequence = 3
assert_true = True

train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

layer_count = 0 #ordering of layers
time_stamp = 0 #use this to indicate when the layer was added

if incrementally_add_layers:
    # ops will give the order of layers
    if dataset_type=='cifar-10':
        iconv_ops = ['conv_1','pool_1','pool_global','fulcon_out']
    elif dataset_type == 'imagenet-100':
        iconv_ops = ['conv_1','pool_1','pool_global','fulcon_out']

else:
    # ops will give the order of layers
    if dataset_type=='cifar-10':
        iconv_ops = ['conv_1','pool_1','conv_2','pool_2','conv_3','conv_4','pool_global','fulcon_out']
    elif dataset_type == 'imagenet-100':
        iconv_ops = ['conv_1','pool_1','conv_2','pool_2',
                     'conv_3','pool_3','conv_4',
                     'conv_5','pool_5','conv_6',
                     'conv_7','pool_global','fulcon_out']

final_2d_output = (1,1)

if dataset_type == 'cifar-10':
    depth_conv = {'conv_1':64,'conv_2':128,'conv_3':256,'conv_4':512}
    conv_1_hyparams = {'weights':[5,5,num_channels,depth_conv['conv_1']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_2_hyparams = {'weights':[5,5,int(depth_conv['conv_1']),depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_3_hyparams = {'weights':[3,3,int(depth_conv['conv_2']),depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_4_hyparams = {'weights':[3,3,int(depth_conv['conv_3']),depth_conv['conv_4']],'stride':[1,1,1,1],'padding':'SAME'}

    pool_1_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}

    if incrementally_add_layers:
        current_final_2d_output = (image_size/pool_1_hyparams['stride'][1], pool_1_hyparams['stride'][2])
        global_kernel_size = [1,int(current_final_2d_output[0]/final_2d_output[0]),
                                                       int(current_final_2d_output[1]/final_2d_output[1]),1]
        global_stride = global_kernel_size if final_2d_output!=(1,1) else [1,1,1,1]
        pool_global_hyparams = {'type':'avg','kernel':global_kernel_size,
                                'stride':global_stride,'padding':'VALID'}

        out_hyparams = {'in':final_2d_output[0]*final_2d_output[1]*depth_conv['conv_1'],'out':num_labels}

    else:
        pool_global_hyparams = {'type':'avg','kernel':[1,8,8,1],'stride':[1,1,1,1],'padding':'VALID'}
        out_hyparams = {'in':final_2d_output[0]*final_2d_output[1]*depth_conv['conv_4'],'out':num_labels}

    hyparams = {
        'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams,
        'conv_3':conv_3_hyparams, 'conv_4':conv_4_hyparams,
        'pool_1': pool_1_hyparams, 'pool_2': pool_1_hyparams,
        'pool_3':pool_1_hyparams,
        'pool_global':pool_global_hyparams,
        'fulcon_out':out_hyparams,
        'final_2d_output':final_2d_output
    }

elif dataset_type == 'imagenet-100':

    depth_conv = {'conv_1':256,'conv_2':512,'conv_3':1024,'conv_4':1024,'conv_5':1024,'conv_6':2048,'conv_7':2048}

    conv_1_hyparams = {'weights':[7,7,num_channels,depth_conv['conv_1']],'stride':[1,2,2,1],'padding':'SAME'}
    conv_2_hyparams = {'weights':[3,3,depth_conv['conv_1'],depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_3_hyparams = {'weights':[3,3,depth_conv['conv_2'],depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_4_hyparams = {'weights':[3,3,depth_conv['conv_3'],depth_conv['conv_4']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_5_hyparams = {'weights':[3,3,depth_conv['conv_4'],depth_conv['conv_5']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_6_hyparams = {'weights':[3,3,depth_conv['conv_5'],depth_conv['conv_6']],'stride':[1,1,1,1],'padding':'SAME'}
    conv_7_hyparams = {'weights':[3,3,depth_conv['conv_6'],depth_conv['conv_7']],'stride':[1,1,1,1],'padding':'SAME'}

    pool_1_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
    pool_3_hyparams = {'type':'avg','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}

    if incrementally_add_layers:
        current_final_2d_output = (image_size / conv_1_hyparams['stride'][1], image_size / conv_1_hyparams['stride'][2])
        pool_global_hyparams = {'type':'avg','kernel':[1,current_final_2d_output[0],current_final_2d_output[1],1],'stride':[1,1,1,1],'padding':'VALID'}
        out_hyparams = {'in':final_2d_output[0]*final_2d_output[1]*depth_conv['conv_1'],'out':num_labels}

    else:
        current_final_2d_output = (7,7)
        global_kernel_size = [1, int(current_final_2d_output[0] / final_2d_output[0]),
                              int(current_final_2d_output[1] / final_2d_output[1]), 1]
        global_stride = global_kernel_size if final_2d_output != (1, 1) else [1, 1, 1, 1]
        pool_global_hyparams = {'type':'avg','kernel':global_kernel_size,'stride':global_stride,'padding':'VALID'}
        out_hyparams = {'in':final_2d_output[0]*final_2d_output[1]*depth_conv['conv_7'],'out':num_labels}

    hyparams = {
        'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams,
        'conv_3': conv_3_hyparams, 'conv_4': conv_4_hyparams,
        'conv_5': conv_5_hyparams, 'conv_6': conv_6_hyparams,
        'conv_7':conv_7_hyparams,
        'pool_1': pool_1_hyparams, 'pool_2': pool_1_hyparams,
        'pool_3': pool_1_hyparams, 'pool_4': pool_1_hyparams,
        'pool_5': pool_1_hyparams,
        'pool_global': pool_global_hyparams,
        'fulcon_out': out_hyparams
    }

conv_depths = {} # store the in and out depth of each convolutional layer
conv_order  = {} # layer order (in terms of convolutional layer) in the full structure

weights,biases = {},{}

print('Hyperparametesr defined ')
print(hyparams)
print()

def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def init_iconvnet():
    global dataset_type

    print('Initializing the iConvNet for %s...'%dataset_type)

    # Convolutional layers
    for op in iconv_ops:
        if 'conv' in op:
                print('\tDefining weights and biases for %s (weights:%s)'%(op,hyparams[op]['weights']))
                print('\t\tWeights:%s'%hyparams[op]['weights'])
                print('\t\tBias:%d'%hyparams[op]['weights'][3])
                weights[op]=tf.Variable(
                    tf.truncated_normal(hyparams[op]['weights'],
                                        stddev=min(0.02,2./(hyparams[op]['weights'][0]*hyparams[op]['weights'][1]))
                                        )
                ,name='w_' + op)
                biases[op] = tf.Variable(tf.constant(np.random.random()*1e-6,shape=[hyparams[op]['weights'][3]]),name='b_'+op)

    # Fully connected classifier
    weights['fulcon_out'] = tf.Variable(tf.truncated_normal(
        [hyparams['fulcon_out']['in'],hyparams['fulcon_out']['out']],
        stddev=2./hyparams['fulcon_out']['in']
    ),name='w_fulcon_out')
    biases['fulcon_out'] = tf.Variable(tf.constant(
        np.random.random()*0.01,shape=[hyparams['fulcon_out']['out']]
    ),name='b_fulcon_out')

    print('Weights for %s initialized with size %d,%d'%(
        'fulcon_out',hyparams['fulcon_out']['in'], num_labels
    ))
    print('Biases for %s initialized with size %d'%(
        'fulcon_out',num_labels
    ))


def append_new_layer_id(layer_id):
    global iconv_ops
    out_id = iconv_ops.pop(-1)
    pool_balance_id = iconv_ops.pop(-1)
    iconv_ops.append(layer_id)
    iconv_ops.append(pool_balance_id)
    iconv_ops.append(out_id)

def update_fulcon_layer():
    global final_2d_output
    '''
    If a convolutional layer (intermediate) is removed. This function will
     Correct the dimention mismatch either by adding more weights or removing
    :param to_d: the depth to be updated to
    :return:
    '''

    last_conv_op = None
    for rev_op in reversed(iconv_ops):
        if 'conv' in rev_op:
            last_conv_op = rev_op
            break

    new_fulcon_in = final_2d_output[0] * final_2d_output[1] * hyparams[last_conv_op]['weights'][3]

    logger.info("Need to update the fulcon_out from %d to %d"%(hyparams['fulcon_out']['in'],new_fulcon_in))

    # need to remove neurons
    if hyparams['fulcon_out']['in']>new_fulcon_in:
        amount_to_rm = hyparams['fulcon_out']['in'] - new_fulcon_in
        new_weights = tf.Variable(tf.slice(weights['fulcon_out'],[0,0],[new_fulcon_in,num_labels]),name='w_fulcon_out')
        tf.variables_initializer([new_weights]).run()
        weights['fulcon_out'] = new_weights

    # need to add neurons
    elif hyparams['fulcon_out']['in']<new_fulcon_in:
        amount_to_add = new_fulcon_in - hyparams['fulcon_out']['in']

        adding_weights = tf.truncated_normal([amount_to_add,num_labels],stddev=0.02)
        new_weights = tf.Variable(tf.concat(0,[weights['fulcon_out'],adding_weights]),name='w_fulcon_out')
        tf.variables_initializer([new_weights]).run()
        weights['fulcon_out'] = new_weights

    # update fulcon_out in
    hyparams['fulcon_out']['in'] = new_fulcon_in


def add_conv_layer(w,stride,conv_id,init_random):
    '''
    Specify the tensor variables and hyperparameters for a convolutional neuron layer. And update conv_ops list
    :param w: Weights of the receptive field
    :param stride: Stried of the receptive field
    :return: None
    '''
    global layer_count,weights,biases,hyparams

    hyparams[conv_id]={'weights':w,'stride':stride,'padding':'SAME'}

    logger.debug('Initializing %s random ...'%conv_id)
    weights[conv_id]= tf.Variable(tf.truncated_normal(w,stddev=2./min(0.02,w[0]*w[1])),name='w_'+conv_id)
    biases[conv_id] = tf.Variable(tf.constant(np.random.random()*0.01,shape=[w[3]]),name='b_'+conv_id)

    tf.variables_initializer([weights[conv_id],biases[conv_id]]).run()

    # add the new conv_op to ordered operation list
    append_new_layer_id(conv_id)

    weight_shape = weights[conv_id].get_shape().as_list()

    for op in reversed(iconv_ops[:iconv_ops.index(conv_id)]):
        if 'conv' in op:
            prev_conv_op = op
            break

    assert prev_conv_op != conv_id


    print('Updating the fulcon layer (if needed)')
    update_fulcon_layer()

    if stride[1]>1 or stride[2]>1:
        raise NotImplementedError

    assert np.all(np.asarray(weight_shape)>0)

def add_pool_layer(ksize,stride,type,pool_id):
    '''
    Specify the hyperparameters for a pooling layer. And update conv_ops
    :param ksize: Kernel size
    :param stride: Stride
    :param type: avg or max
    :return: None
    '''
    global layer_count,hyparams
    hyparams[pool_id] = {'type':type,'kernel':ksize,'stride':stride,'padding':'SAME'}
    append_new_layer_id(pool_id)


def get_logits(dataset):
    global final_2d_output,current_final_2d_output
    outputs = []
    logger.info('Current set of operations: %s'%iconv_ops)
    outputs.append(dataset)
    logger.debug('Received data for X(%s)...'%outputs[-1].get_shape().as_list())

    logger.info('Performing the specified operations ...')

    #need to calculate the output according to the layers we have
    for op in iconv_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,hyparams[op]['weights'],hyparams[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(outputs[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            outputs.append(tf.nn.conv2d(outputs[-1], weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding']))
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(outputs[-1].get_shape().as_list(),biases[op].get_shape().as_list()))
            outputs[-1] = tf.nn.relu(outputs[-1] + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,outputs[-1].get_shape().as_list()))

        if 'pool' in op:

            if op =='pool_global':
                shape = outputs[-1].get_shape().as_list()
                current_final_2d_output = (int(shape[1]/final_2d_output[0]), int(shape[2]/final_2d_output[1]))

                hyparams['pool_global']['kernel'] = [1, current_final_2d_output[0], current_final_2d_output[1], 1]
                if current_final_2d_output==(1,1):
                    hyparams['pool_global']['stride'] = [1,1,1,1]
                else:
                    hyparams['pool_global']['stride'] = [1, current_final_2d_output[0], current_final_2d_output[1], 1]

            logger.debug('\t%s Pooling (%s) with Kernel:%s Stride:%s' % (
                hyparams[op]['type'], op, hyparams[op]['kernel'], hyparams[op]['stride']))

            if hyparams[op]['type']=='max':
                outputs.append(tf.nn.max_pool(outputs[-1],ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],
                                              padding=hyparams[op]['padding']))
            elif hyparams[op]['type']=='avg':
                outputs.append(tf.nn.avg_pool(outputs[-1], ksize=hyparams[op]['kernel'], strides=hyparams[op]['stride'],
                                              padding=hyparams[op]['padding']))
            logger.debug('\t\tX after %s:%s'%(op,outputs[-1].get_shape().as_list()))

        if op=='loc_res_norm':
            print('\tLocal Response Normalization')
            outputs.append(tf.nn.local_response_normalization(outputs[-1], depth_radius=3, bias=None, alpha=1e-2, beta=0.75))

        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = outputs[-1].get_shape().as_list()

    logger.debug("Req calculation, Actual calculation: (%d,%d), (%d,%d)"
                 %(final_2d_output[0],final_2d_output[1],current_final_2d_output[0],current_final_2d_output[1]))
    rows = shape[0]

    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams['fulcon_out']['in'])))
    reshaped_output = tf.reshape(outputs[-1], [rows,hyparams['fulcon_out']['in']])

    outputs.append(tf.matmul(reshaped_output, weights['fulcon_out']) + biases['fulcon_out'])

    return outputs


def calc_loss(logits,labels):
    # Training computation.
    if include_l2_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss


def calc_loss_vector(logits,labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits, labels)


def optimize_func(loss,global_step):
    # Optimizer.
    if decay_learning_rate:
        learning_rate = tf.maximum(tf.constant(start_lr*0.1),
                                   tf.train.exponential_decay(start_lr, global_step,decay_steps=1,decay_rate=0.9)
                                   )
    else:
        learning_rate = start_lr

    # Using SGD as the optimizer caused the incrementally adding layers to result in higher accuracy
    # compared to starting with all the layers
    # Using SGD with momentum alleviates this
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer,learning_rate



def inc_global_step(global_step):
    return global_step.assign(global_step+1)


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction

def predict_with_dataset(dataset):
    prediction = tf.nn.softmax(get_logits(dataset)[-1])
    return prediction

def get_weight_stats():
    global weights
    stats=dict()
    for op,w in weights.items():
        if 'conv' in op:
            w_nd = w.eval()
            stats[op]= {'min':np.min(w_nd),'max':np.max(w_nd),'mean':np.mean(w_nd),'stddev':np.std(w_nd)}

    return stats


def find_max_activations_for_layer(activations):
    # this should return the id of the image that had max activation for all feature maps

    # this matrix of size b x d
    max_activations_images = tf.reduce_max(activations,axis=[1,2])
    img_id_with_max_activation = tf.argmax(max_activations_images, axis=0)
    return max_activations_images,img_id_with_max_activation

def deconv_featuremap_with_data(layer_id,featuremap_id,tf_selected_dataset,guided_backprop=False):
    global weights,biases

    pool_switches = {}
    activation_masks = {} # used for guided_backprop
    outputs_fwd = []
    logger.info('Current set of operations: %s'%iconv_ops)
    outputs_fwd.append(tf_selected_dataset)
    logger.debug('Received data for X(%s)...'%outputs_fwd[-1].get_shape().as_list())

    logger.info('Performing the forward pass ...')

    output_fwd_for_layer = None
    #need to calculate the output according to the layers we have
    for op in iconv_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,hyparams[op]['weights'],hyparams[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(outputs_fwd[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            outputs_fwd.append(tf.nn.conv2d(outputs_fwd[-1], weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding']))
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(outputs_fwd[-1].get_shape().as_list(),biases[op].get_shape().as_list()))

            # should happend before RELU application
            if guided_backprop:
                activation_masks[op] = tf.greater(outputs_fwd[-1],tf.constant(0,dtype=tf.float32))
                assert activation_masks[op].get_shape().as_list() == outputs_fwd[-1].get_shape().as_list()

            outputs_fwd[-1] = tf.nn.relu(outputs_fwd[-1] + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,outputs_fwd[-1].get_shape().as_list()))

            if op==layer_id:
                output_fwd_for_layer = outputs_fwd[-1]

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,hyparams[op]['kernel'],hyparams[op]['stride']))
            pool_out,switch = tf.nn.max_pool_with_argmax(outputs_fwd[-1],ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding'])
            outputs_fwd.append(pool_out)
            pool_switches[op] = switch
            logger.debug('\t\tX after %s:%s'%(op,outputs_fwd[-1].get_shape().as_list()))

        if 'fulcon' in op:
            break

    shape = outputs_fwd[-1].get_shape().as_list()
    rows = shape[0]

    print('Unwrapping last convolution layer %s to %s hidden layer' % (shape, (rows, hyparams['fulcon_out']['in'])))
    reshaped_output = tf.reshape(outputs_fwd[-1], [rows, hyparams['fulcon_out']['in']])

    outputs_fwd.append(tf.matmul(reshaped_output, weights['fulcon_out']) + biases['fulcon_out'])

    logger.info('Performing the backward pass ...\n')
    logger.debug('\tInput Size (Non-Zeroed): %s',str(outputs_fwd[-1].get_shape().as_list()))

    # b h w d parameters of the required layer
    # b - batch size, h - height, w - width, d - number of filters
    b,h,w,d = output_fwd_for_layer.get_shape().as_list()

    # outputs[-1] will have the required activation
    # will be of size b x 1 x 1

    # we create a tensor from the activations of the layer which only has non-zeros
    # for the selected feature map (layer_activations_2)
    layer_activations = tf.transpose(output_fwd_for_layer,[3,0,1,2])
    layer_indices = tf.constant([[featuremap_id]])
    layer_updates = tf.expand_dims(layer_activations[featuremap_id,:,:,:],0)
    layer_activations_2 = tf.scatter_nd(layer_indices,layer_updates,tf.constant(layer_activations.get_shape().as_list()))
    layer_activations_2 = tf.transpose(layer_activations_2,[1,2,3,0])
    assert output_fwd_for_layer.get_shape().as_list() == layer_activations_2.get_shape().as_list()

    # single out only the maximally activated neuron and set the zeros
    argmax_indices = tf.argmax(tf.reshape(layer_activations_2,[b,h*w*d]),axis=1)
    batch_range = tf.range(b,dtype=tf.int32)
    nonzero_indices = tf.stack([batch_range,tf.to_int32(argmax_indices)],axis=1)
    updates = tf.gather_nd(tf.reshape(layer_activations_2,[b,h*w*d]),nonzero_indices)

    # OBSOLETE At the Moment
    # this will be of size layer_id (some type of b x h x w x d)
    dOut_over_dh = tf.gradients(outputs_fwd[-1],output_fwd_for_layer)[0]
    deriv_updates = tf.gather_nd(tf.reshape(dOut_over_dh,[b,h*w*d]),nonzero_indices)

    logger.debug('\tNon-zero indices shape: %s',nonzero_indices.get_shape().as_list())
    logger.debug('\tNon-zero updates shape: %s',updates.get_shape().as_list())
    logger.debug('\tdOut/dh shape: %s',dOut_over_dh.get_shape().as_list())

    # OBSOLETE
    # Creating the new gradient tensor (of size: b x w x h x d) for deconv
    # with only the gradient for highest activation of given feature map ID non-zero and rest set to zero
    zeroed_derivatives = tf.scatter_nd(nonzero_indices,updates,tf.constant([b,h*w*d],dtype=tf.int32))
    zeroed_derivatives = tf.reshape(zeroed_derivatives,[b,h,w,d])

    outputs_bckwd = [zeroed_derivatives] # this will be the output of the previous layer to layer_id
    prev_op_index = iconv_ops.index(layer_id)

    logger.debug('Input Size (Zeroed): %s',str(outputs_bckwd[-1].get_shape().as_list()))

    for op in reversed(iconv_ops[:prev_op_index+1]):
        if 'conv' in op:

            # Deconvolution
            logger.debug('\tDeConvolving (%s) With Weights:%s Stride:%s'%(op,weights[op].get_shape().as_list(),hyparams[op]['stride']))
            logger.debug('\t\tX before deconvolution:%s'%(outputs_bckwd[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())

            output_shape = outputs_bckwd[-1].get_shape().as_list()
            output_shape[1] *= hyparams[op]['stride'][1]
            output_shape[2] *= hyparams[op]['stride'][2]
            output_shape[3] = hyparams[op]['weights'][2]
            logger.debug('\t\tExpected output shape: %s',output_shape)
            outputs_bckwd.append(
                    tf.nn.conv2d_transpose(outputs_bckwd[-1], filter=weights[op], strides=hyparams[op]['stride'],
                                           padding=hyparams[op]['padding'],output_shape=tf.constant(output_shape))
            )

            logger.debug('\t\tX after %s:%s'%(op,outputs_bckwd[-1].get_shape().as_list()))

        if 'pool' in op:

            # find previous conv_op
            previous_conv_op = None
            for before_op in reversed(iconv_ops[:iconv_ops.index(op)+1]):
                if 'conv' in before_op:
                    previous_conv_op = before_op
                    break

            logger.debug('\tDetected previous conv op %s',previous_conv_op)

            # Unpooling operation and Rectification
            logger.debug('\tUnPooling (%s) with Kernel:%s Stride:%s'%(op,hyparams[op]['kernel'],hyparams[op]['stride']))
            logger.debug('\t\tInput shape: %s',outputs_bckwd[-1].get_shape().as_list())

            output_shape = outputs_bckwd[-1].get_shape().as_list()
            output_shape[1] *= hyparams[op]['stride'][1]
            output_shape[2] *= hyparams[op]['stride'][2]

            logger.debug('\t\tExpected output shape: %s',output_shape)
            # Unpooling

            # Switch variable returns an array of size b x h x w x d. But only provide flattened indices
            # Meaning that if you have an output of size 4x4 it will flatten it to a 16 element long array

            # we're goin go make a batch_range which is like [0,0,...,0,1,1,...,1,...]
            # so each unique number will have (h/stride * w/stride * d) elements
            # first it will be of shape b x h/stride x w/stride x d
            # but then we reshape it to b x (h/stride * w/stride * d)
            tf_switches = pool_switches[op]
            tf_batch_range = tf.reshape(tf.range(b,dtype=tf.int32),[b,1,1,1])
            tf_ones_mask = tf.ones_like(tf_switches,dtype=tf.int32)
            tf_multi_batch_range = tf_ones_mask * tf_batch_range

            # here we have indices that looks like b*(h/stride)*(w/stride) x 2
            tf_indices = tf.stack([tf.reshape(tf_multi_batch_range,[-1]),tf.reshape(tf.to_int32(tf_switches),[-1])],axis=1)

            updates = tf.reshape(outputs_bckwd[-1],[-1])

            ref = tf.Variable(tf.zeros([b,output_shape[1]*output_shape[2]*output_shape[3]],dtype=tf.float32),dtype=tf.float32,name='ref_'+op,trainable=False)

            session.run(tf.variables_initializer([ref]))

            updated_unpool = tf.scatter_nd(tf.to_int32(tf_indices),updates,tf.constant([b,output_shape[1]*output_shape[2]*output_shape[3]]),name='updated_unpool_'+op)

            outputs_bckwd.append(tf.reshape(updated_unpool,[b,output_shape[1],output_shape[2],output_shape[3]]))

            # should happen before RELU
            if guided_backprop and previous_conv_op is not None:
                logger.info('Output-bckwd: %s',outputs_bckwd[-1].get_shape().as_list())
                logger.info('Activation mask %s',activation_masks[previous_conv_op].get_shape().as_list())
                assert outputs_bckwd[-1].get_shape().as_list() == activation_masks[previous_conv_op].get_shape().as_list()
                outputs_bckwd[-1] = outputs_bckwd[-1] * tf.to_float(activation_masks[previous_conv_op])

            outputs_bckwd[-1] = tf.nn.relu(outputs_bckwd[-1])

            logger.debug('\t\tX after %s:%s'%(op,outputs_bckwd[-1].get_shape().as_list()))

    return outputs_fwd,outputs_bckwd

def visualize_with_deconv(session,layer_id,all_x,guided_backprop=False):
    global logger, weights, biases
    '''
    DECONV works the following way.
    # Pick a layer
    # Pick a subset of feature maps or all the feature maps in the layer (if small)
    # For each feature map
    #     Pick the n images that maximally activate that feature map
    #     For each image
    #          Do back propagation for the given activations from that layer until the pixel input layer
    '''

    selected_featuremap_ids = list(np.random.randint(0,depth_conv[layer_id],(20,)))

    examples_per_featuremap = 10
    images_for_featuremap = {} # a dictionary with featuremmap_id : an ndarray with size num_of_images_per_featuremap x image_size
    mean_activations_for_featuremap = {} # this is a dictionary containing featuremap_id : [list of mean activations for each image in order]

    layer_index = iconv_ops.index(layer_id)
    tf_deconv_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_selected_image = tf.placeholder(tf.float32, shape=(examples_per_featuremap, image_size, image_size, num_channels))

    tf_activations = get_logits(tf_deconv_dataset)
    # layer_index+1 because we have input as 0th index
    tf_max_activations, tf_img_ids = find_max_activations_for_layer(tf_activations[layer_index+1])

    # activations for batch of data for a layer of size w(width),h(height),d(depth) will be = b x w x h x d
    # reduce this to b x d by using reduce sum
    image_shape = all_x.shape[1:]
    for batch_id in range(all_x.shape[0]//batch_size):

        batch_data = all_x[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]

        # max_activations b x d, img_ids_for_max will be 1 x d
        max_activations,img_ids_for_max = session.run([tf_max_activations,tf_img_ids],feed_dict={tf_deconv_dataset:batch_data})
        if batch_id==0:
            logger.debug('Max activations for batch %d size: %s',batch_id,str(max_activations.shape))
            logger.debug('Image ID  for batch %d size: %s',batch_id,str(img_ids_for_max.shape))

        for d_i in range(img_ids_for_max.shape[0]):

            # we only run this for selected set of featurmaps
            if d_i not in selected_featuremap_ids:
                continue

            img_id = np.asscalar(img_ids_for_max[d_i])

            if d_i==selected_featuremap_ids[1]:
                logger.debug('Found %d image_id for depth %d',img_id,d_i)

            if d_i not in mean_activations_for_featuremap:
                mean_activations_for_featuremap[d_i] = [np.asscalar(max_activations[img_id,d_i])]
                images_for_featuremap[d_i] = np.reshape(batch_data[img_id,:,:,:],(1,image_shape[0],image_shape[1],image_shape[2]))

                # appending the maximum
                reshp_img = np.reshape(batch_data[img_id, :, :, :], (1, image_shape[0], image_shape[1], image_shape[2]))
                images_for_featuremap[d_i] = np.append(images_for_featuremap[d_i], reshp_img, axis=0)
                mean_activations_for_featuremap[d_i].append(np.asscalar(max_activations[img_id, d_i]))

            else:
                if len(mean_activations_for_featuremap[d_i])>= examples_per_featuremap:
                    # delete the minimum
                    # if minimum is less then the oncoming one
                    if np.min(mean_activations_for_featuremap[d_i]) < np.asscalar(max_activations[img_id,d_i]):
                        min_idx = np.asscalar(np.argmin(np.asarray(mean_activations_for_featuremap[d_i])))
                        if d_i==selected_featuremap_ids[1]:
                            logger.debug('Mean activations: %s',mean_activations_for_featuremap[d_i])
                            logger.debug('\tFound minimum activation with %.2f at index %d',np.min(mean_activations_for_featuremap[d_i]),min_idx)

                        # deleting the minimum
                        del mean_activations_for_featuremap[d_i][min_idx]
                        images_for_featuremap[d_i] = np.delete(images_for_featuremap[d_i],[min_idx],axis=0)

                        # appending the maximum
                        reshp_img = np.reshape(batch_data[img_id,:,:,:],(1,image_shape[0],image_shape[1],image_shape[2]))
                        images_for_featuremap[d_i] = np.append(images_for_featuremap[d_i],reshp_img,axis=0)
                        mean_activations_for_featuremap[d_i].append(np.asscalar(max_activations[img_id,d_i]))

    logger.info('Size of image set for feature map: %s',str(len(mean_activations_for_featuremap[selected_featuremap_ids[0]])))

    # TODO: run the following command for all the selected featuremap_id
    all_deconv_outputs = []
    all_images = []
    for d_i in selected_featuremap_ids:
        tf_fwd_outputs, tf_bck_outputs = deconv_featuremap_with_data(layer_id,d_i,tf_selected_image,guided_backprop)
        fwd_outputs, deconv_outputs = session.run([tf_fwd_outputs,tf_bck_outputs],
                                                  feed_dict={tf_selected_image:images_for_featuremap[d_i]})
        all_deconv_outputs.append(deconv_outputs[-1])
        all_images.append(images_for_featuremap[d_i])
    return all_deconv_outputs, all_images

# this method takes either the full dataset or a chunk
# if the dataset is too large and train with that

tf_train_logits,tf_train_loss,tf_train_pred,tf_optimize,tf_inc_gstep = None,None,None,None,None
momentum = None

def train_with_dataset(session, tf_train_dataset, tf_train_labels, chunk_dataset, chunk_labels):
    '''
    This method takes a set of training data and labels and train the model
    on that data. This method is used as a way to accomodate both loading the
    data at once and loading as small chunks (imagent)

    :param session: Tensorflow session
    :param tf_train_dataset: Training dataset placeholder
    :param tf_train_labels: Training labels placeholder
    :param chunk_dataset: Actual dataset (ndarray)
    :param chunk_labels:  Actual one-hot labels (ndarray)
    :return: None

    '''
    global tf_train_logits,tf_train_loss,tf_train_pred,tf_optimize,tf_inc_gstep
    global logger

    logger.debug('Loaded data of size: %s, %s', chunk_dataset.shape[0],chunk_labels.shape[0])

    if tf_train_logits is None or tf_train_loss is None or tf_train_pred is None or \
                    tf_optimize is None or tf_inc_gstep is None:
        tf_train_logits = get_logits(tf_train_dataset)[-1]
        tf_train_loss = calc_loss(tf_train_logits, tf_train_labels)
        tf_train_pred = predict_with_logits(tf_train_logits)
        tf_optimize = optimize_func(tf_train_loss, global_step)

        # required for the momentum optimizer
        tf.global_variables_initializer().run()

    train_accuracy_log = []
    train_loss_log = []

    for batch_id in range(ceil(chunk_dataset.shape[0] // batch_size) - 1):
        batch_data = chunk_dataset[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
        batch_labels = chunk_labels[batch_id * batch_size:(batch_id + 1) * batch_size, :]

        train_feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, (_, updated_lr), predictions = session.run(
            [tf_train_logits, tf_train_loss, tf_optimize, tf_train_pred], feed_dict=train_feed_dict
        )

        train_accuracy_log.append(accuracy(predictions, batch_labels))
        train_loss_log.append(l)

    return {'train_accuracy':np.mean(train_accuracy_log),
            'train_loss':np.mean(train_loss_log),
            'updated_lr':updated_lr}


if __name__=='__main__':
    global logger

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",["output_dir="])
    except getopt.GetoptError as err:
        print('<filename>.py --output_dir=')

    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--output_dir':
                output_dir = arg

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    # Value logger will log info used to calculate policies
    training_stat_logger = logging.getLogger('stat_logger')
    training_stat_logger.setLevel(logging.INFO)
    statfileHandler = logging.FileHandler(output_dir+os.sep+'training_log.log', mode='w')
    statfileHandler.setFormatter(logging.Formatter('%(message)s'))
    training_stat_logger.addHandler(statfileHandler)

    # Value logger will log info used to calculate policies
    cnnnet_logger = logging.getLogger('cnnnet_logger')
    cnnnet_logger.setLevel(logging.INFO)
    cnnfileHandler = logging.FileHandler(output_dir + os.sep + 'cnn_log.log', mode='w')
    cnnfileHandler.setFormatter(logging.Formatter('%(message)s'))
    cnnnet_logger.addHandler(cnnfileHandler)

    cnnnet_logger.info('CNN Hyperparameters')
    for k,v in hyparams.items():
        cnnnet_logger.info('%s: %s',k,v)
    cnnnet_logger.info('')

    cnnnet_logger.info('Batch size: %d',batch_size)
    cnnnet_logger.info('Incrementally Add Layers: %s',incrementally_add_layers)
    cnnnet_logger.info('Learning rate: %.5f',start_lr)
    cnnnet_logger.info('Decay Learning Rate: %s',decay_learning_rate)
    cnnnet_logger.info('Dropout: %.3f',dropout_rate)
    cnnnet_logger.info('Use Dropout: %s',use_dropout)
    cnnnet_logger.info('Include L2 Loss: %s',include_l2_loss)
    cnnnet_logger.info('Beta: %.3f',beta)
    cnnnet_logger.info('Validation Frequency: %d',validation_frequency)
    cnnnet_logger.info('Valid Drop Cap: %d',valid_drop_cap)
    cnnnet_logger.info('No Improvement Cap: %d',no_improvement_cap)
    cnnnet_logger.info('Incrementing Layer Frequency : %d',incrementing_frequence)


    cnnnet_logger.info('CNN Operations in order (Beginning)')
    cnnnet_logger.info('%s\n',iconv_ops)

    with open(output_dir + os.sep + 'ops_hyperparameters.pickle', 'wb') as f:
        pickle.dump({'depth_conv':depth_conv,'op_list':iconv_ops,'hyperparameters':hyparams}, f, pickle.HIGHEST_PROTOCOL)

    graph = tf.Graph()

    valid_accuracy_log = []
    test_accuracy_log = []

    with tf.Session(graph=graph,config = tf.ConfigProto(allow_soft_placement=True)) as session:

        #global step is used to decay the learning rate
        global_step = tf.Variable(0, trainable=False,name='global_step')

        logger.info('Input data defined...\n')

        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size, image_size, num_channels),
                                          name='train_dataset')
        tf_train_labels = tf.placeholder(tf.float32,
                                         shape=(batch_size, num_labels),
                                         name='train_labels')

        # Valid data
        tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        init_iconvnet() #initialize the initial conv net
        op_variables = [v.name for v in tf.trainable_variables()]
        logger.info(op_variables)
        logger.info('='*80)
        tf_inc_gstep = inc_global_step(global_step)

        # valid predict function
        pred_valid = predict_with_dataset(tf_valid_dataset)
        pred_test = predict_with_dataset(tf_test_dataset)

        tf.global_variables_initializer().run()

        uninit_vars = session.run(tf.report_uninitialized_variables(list(weights.values()).extend(biases.values()), name='report_uninitialized_variables'))
        if len(uninit_vars)>0:
            raise AssertionError

        # initial loading
        if dataset_type=='imagenet-100':
            chunk_size = batch_size * 50
            train_size_clipped = train_size
            chunks_in_train = train_size_clipped // chunk_size
            if abs(train_size_clipped - chunks_in_train * chunk_size) < 5 * batch_size:
                train_size_clipped = chunks_in_train * chunk_size
                print('Clipped size is slightly off from factor of chunk size. Using %d as train size' % train_size_clipped)

            print('Running for %d data points' % train_size_clipped)

            train_dataset_filename = 'imagenet_small' + os.sep + 'imagenet_small_train_dataset'
            train_label_filename = 'imagenet_small' + os.sep + 'imagenet_small_train_labels'

            valid_dataset_fname = 'imagenet_small' + os.sep + 'imagenet_small_valid_dataset'
            valid_label_fname = 'imagenet_small' + os.sep + 'imagenet_small_valid_labels'

            fp1 = np.memmap(valid_dataset_fname, dtype=np.float32, mode='r',
                            offset=np.dtype('float32').itemsize * 0,
                            shape=(valid_size, image_size, image_size, num_channels))
            fp2 = np.memmap(valid_label_fname, dtype=np.int32, mode='r',
                            offset=np.dtype('int32').itemsize * 0, shape=(valid_size, 1))
            valid_dataset = fp1[:, :, :, :]
            valid_labels = fp2[:]

            valid_dataset, valid_labels = load_data.reformat_data_imagenet_with_memmap_array(
                valid_dataset, valid_labels, silent=True
            )

            to_add_ops = ['conv_2', 'pool_2',
                          'conv_3', 'pool_3', 'conv_4',
                          'conv_5', 'pool_5', 'conv_6',
                          'conv_7']

        elif dataset_type=='cifar-10':
            train_size_clipped = batch_size*(train_size//batch_size)
            to_add_ops = ['conv_2', 'pool_2', 'conv_3', 'conv_4']

        chunk_train_dataset, chunk_train_labels = None, None

        weight_saver = tf.train.Saver(weights)
        bias_saver = tf.train.Saver(biases)

        max_valid_accuracy = 0.0
        valid_drop_count = 0
        no_improvement_count = 0
        for epoch in range(31):
            filled_size = 0

            train_chunk_accuracy_log = []
            train_chunk_error_log = []

            # changing the structure of the network
            if len(to_add_ops)>0 and epoch>0 and \
                                    (epoch+1)%incrementing_frequence==0 and incrementally_add_layers:

                # save weights
                weight_saver.save(session, output_dir + os.sep + 'cnn-weights', epoch+1)
                bias_saver.save(session, output_dir + os.sep + 'cnn-biases', epoch+1)

                to_conv_id = to_add_ops[0]
                logger.info('Adding convolution layer %s',to_conv_id)
                add_conv_layer(hyparams[to_conv_id]['weights'],hyparams[to_conv_id]['stride'],to_conv_id,True)
                del to_add_ops[0]

                if len(to_add_ops)>0 and 'pool' in to_add_ops[0]:
                    to_pool_id = to_add_ops[0]
                    logger.info('Adding pooling layer %s', to_conv_id)
                    add_pool_layer(hyparams[to_pool_id]['kernel'],hyparams[to_pool_id]['stride'],hyparams[to_pool_id]['type'],to_pool_id)
                    del to_add_ops[0]

                cnnnet_logger.info('Epoch: %d, CNN Operations changed to: %s',epoch, iconv_ops)
                # update logits after adding new layer
                get_logits(tf_valid_dataset)

                # here we make a few checks
                # first check is all the weights and biases of the ops are trainable
                op_variables = [v.name  for v in tf.trainable_variables()]
                logger.info('Trainable Variables: %s',op_variables)

                for tmp_op in iconv_ops:
                    if 'conv' in tmp_op or 'fulcon' in tmp_op:
                        trainable_error_weights, trainable_error_biases = True, True
                        logger.info('Checking if variables of %s are trainable',tmp_op)
                        for v in op_variables:
                            if v.startswith('w_'+tmp_op):
                                trainable_error_weights = False
                                break
                        for v in op_variables:
                            if v.startswith('b_'+tmp_op):
                                trainable_error_biases = False
                                break

                        if trainable_error_weights or trainable_error_biases:
                            raise AssertionError


                # second check of the size of weights fulcon_out  == final_2d_ouput[0] * [1] * depth[last_conv]
                last_conv_op = None
                for tmp_op in iconv_ops:
                    if 'conv' in tmp_op:
                        last_conv_op = tmp_op
                logger.info('Got fulcon_out size: %d',weights['fulcon_out'].get_shape().as_list()[0])
                assert weights['fulcon_out'].get_shape().as_list()[0] == final_2d_output[0]*final_2d_output[1]*depth_conv[last_conv_op]
                assert hyparams['fulcon_out']['in'] == final_2d_output[0]*final_2d_output[1]*depth_conv[last_conv_op]

            # need to have code to run the training for all the data points
            # we use a while loop to support loading data at once and as chunks both
            # 1 loop = 1 traverse through the full dataset (1 training epoch)
            while filled_size < train_size_clipped:

                if dataset_type == 'imagenet-100':

                    start_memmap, end_memmap = load_data.get_next_memmap_indices((train_dataset_filename, train_label_filename),
                                                                       chunk_size, train_size_clipped)
                    # Loading data from memmap
                    logger.info('Processing files %s,%s' % (train_dataset_filename, train_label_filename))
                    logger.debug('\tFrom index %d to %d',start_memmap,end_memmap)
                    fp1 = np.memmap(train_dataset_filename, dtype=np.float32, mode='r',
                                    offset=np.dtype('float32').itemsize * image_size * image_size * num_channels * start_memmap,
                                    shape=(end_memmap - start_memmap, image_size, image_size, num_channels))

                    fp2 = np.memmap(train_label_filename, dtype=np.int32, mode='r',
                                    offset=np.dtype('int32').itemsize * 1 * start_memmap,
                                    shape=(end_memmap - start_memmap, 1))

                    chunk_train_dataset = fp1[:, :, :, :]
                    chunk_train_labels = fp2[:]

                    chunk_train_dataset, chunk_train_labels = load_data.reformat_data_imagenet_with_memmap_array(
                        chunk_train_dataset,chunk_train_labels,silent=True
                    )

                elif dataset_type == 'cifar-10':
                    data_filename = '..'+os.sep+'data' + os.sep + 'cifar-10.pickle'

                    if chunk_train_labels is None and chunk_train_labels is None:
                        (train_dataset, train_labels), \
                        (valid_dataset, valid_labels), \
                        (test_dataset, test_labels) = load_data.reformat_data_cifar10(data_filename)

                        chunk_train_dataset,chunk_train_labels = train_dataset,train_labels

                train_stats = train_with_dataset(session,tf_train_dataset,tf_train_labels,chunk_train_dataset,chunk_train_labels)
                train_chunk_accuracy_log.append(train_stats['train_accuracy'])
                train_chunk_error_log.append(train_stats['train_loss'])
                filled_size += chunk_train_dataset.shape[0]

                training_stat_logger.info('#Training statistics for Epoch %d, Data points (%d-%d)',epoch,filled_size-chunk_train_dataset.shape[0],filled_size)
                training_stat_logger.info('#Mean Loss, Mean Accuracy')
                training_stat_logger.info('%.4f, %.4f\n',train_stats['train_loss'],train_stats['train_accuracy'])

            logger.info('\tGlobal Step %d' % global_step.eval())
            logger.info('\tMean loss at epoch %d: %.5f' % (epoch, np.mean(train_chunk_error_log)))
            logger.debug('\tLearning rate: %.5f'%train_stats['updated_lr'])
            logger.info('\tMinibatch accuracy: %.2f%%\n' % np.mean(train_chunk_accuracy_log))

            training_stat_logger.info('\n# Training statistics for Epoch %d',epoch)
            training_stat_logger.info('#Mean loss, Mean accuracy, Learning rate')
            training_stat_logger.info('%.4f,%.4f,%.6f\n',np.mean(train_chunk_error_log),np.mean(train_chunk_accuracy_log),train_stats['updated_lr'])

            train_accuracy_log = []
            train_loss_log = []

            if (epoch +1) % validation_frequency == 0:
                for valid_batch_id in range(ceil(valid_size//batch_size)-1):
                    # validation batch
                    batch_valid_data = valid_dataset[(valid_batch_id)*batch_size:(valid_batch_id+1)*batch_size, :, :, :]
                    batch_valid_labels = valid_labels[(valid_batch_id)*batch_size:(valid_batch_id+1)*batch_size, :]

                    valid_feed_dict = {tf_valid_dataset:batch_valid_data,tf_valid_labels:batch_valid_labels}
                    valid_predictions = session.run([pred_valid],feed_dict=valid_feed_dict)
                    valid_accuracy_log.append(accuracy(valid_predictions[0],batch_valid_labels))

                # only cifar-10 has a test dataset
                if dataset_type=='cifar-10':
                    for test_batch_id in range(ceil(test_size//batch_size)-1):
                        # test batch
                        batch_test_data = test_dataset[(test_batch_id)*batch_size:(test_batch_id+1)*batch_size, :, :, :]
                        batch_test_labels = test_labels[(test_batch_id)*batch_size:(test_batch_id+1)*batch_size, :]

                        test_feed_dict = {tf_test_dataset:batch_test_data,tf_test_labels:batch_test_labels}
                        test_predictions = session.run([pred_test],feed_dict=test_feed_dict)
                        test_accuracy_log.append(accuracy(test_predictions[0],batch_test_labels))

                logger.info('\n==================== Epoch: %d ====================='%epoch)
                logger.info('\tGlobal step: %d'%global_step.eval())
                logger.debug('\tCurrent Ops: %s'%iconv_ops)
                mean_valid_accuracy = np.mean(valid_accuracy_log)
                mean_test_accuracy = np.mean(test_accuracy_log) if len(test_accuracy_log)>0 else 0.0
                logger.info('\tMean Valid accuracy: %.2f%%' %mean_valid_accuracy)
                if dataset_type == 'cifar-10':
                    logger.info('\tMean Test accuracy: %.2f%%\n' %mean_test_accuracy)

                valid_accuracy_log = []
                test_accuracy_log = []

                if max_valid_accuracy>mean_valid_accuracy:
                    valid_drop_count += 1
                    logger.info('Incrementing valid_drop to %d',valid_drop_count)
                else:
                    logger.info('Resetting valid_drop and no_improvement variables')
                    valid_drop_count = 0
                    no_improvement_count = 0

                if valid_drop_count>valid_drop_cap:
                    _ = session.run([tf_inc_gstep])
                    no_improvement_count += 1

                if no_improvement_count>no_improvement_cap:
                    logger.info('Validation accuracy has not improved in %d epochs',no_improvement_count)
                    logger.info('Breaking the loop')
                    break

                if mean_valid_accuracy>max_valid_accuracy:
                    max_valid_accuracy = mean_valid_accuracy

                training_stat_logger.info('\n#Validation and Test accuracies for epoch %d',epoch)
                training_stat_logger.info('%.2f,%.2f',mean_valid_accuracy,mean_test_accuracy)

                if save_weights_periodic and (epoch+1)%save_period==0:
                    weight_saver.save(session, output_dir + os.sep + 'cnn-weights', epoch + 1)
                    bias_saver.save(session, output_dir + os.sep + 'cnn-biases', epoch + 1)

        weight_saver.save(session,output_dir+os.sep+'cnn-weights',epoch+1)
        bias_saver.save(session, output_dir + os.sep + 'cnn-biases', epoch+1)

        del chunk_train_dataset,chunk_train_labels
        # Visualization
        '''layer_ids = [op for op in iconv_ops if 'conv' in op and op!='conv_1']

        backprop_feature_dir = output_dir+os.sep+backprop_feature_dir
        if not os.path.exists(backprop_feature_dir):
            os.makedirs(backprop_feature_dir)


        rotate_required = True if dataset_type=='cifar-10' else False # cifar-10 training set has rotated images so we rotate them to original orientation

        for lid in layer_ids:
            all_deconvs,all_images = visualize_with_deconv(session,lid,valid_dataset,False)
            d_i = 0
            for deconv_di, images_di in zip(all_deconvs,all_images):
                local_dir = backprop_feature_dir + os.sep + lid + '_' + str(d_i)
                if not os.path.exists(local_dir):
                    os.mkdir(local_dir)

                # saving deconv images
                for img_i in range(deconv_di.shape[0]):
                    if rotate_required:
                        local_img = (deconv_di[img_i,:,:,:]-np.min(deconv_di[img_i,:,:,:])).astype('uint16')
                        local_img = rotate(local_img,270)
                    else:
                        local_img = deconv_di[img_i,:,:,:]
                    imsave(local_dir + os.sep + 'deconv_' + lid +'_'+str(d_i)+'_'+str(img_i)+'.png', local_img)

                # saving original images
                for img_i in range(images_di.shape[0]):
                    if rotate_required:
                        images_di[img_i,:,:,:] = images_di[img_i,:,:,:] - np.min(images_di[img_i,:,:,:])
                        images_di[img_i,:,:,:] = images_di[img_i,:,:,:]*255.0/np.max(images_di[img_i,:,:,:])
                        local_img = images_di[img_i,:,:,:].astype('uint16')

                        local_img = rotate(local_img,270)
                    else:
                        local_img = images_di[img_i,:,:,:]
                    imsave(local_dir + os.sep + 'image_' + lid +'_'+str(d_i)+'_'+str(img_i)+'.png', local_img)

                d_i += 1'''
