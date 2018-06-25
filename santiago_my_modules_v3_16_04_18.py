import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

#import time
#from sklearn.model_selection import train_test_split

from tensorflow.python.framework import ops

#from cnn_utils import *

def split_dataset(X, Y, validation_folder, test_folder, files_per_folder):
    """
    Splits the dataset into train, test and validation set.
    
    Arguments:
    X -- array with the whole dataset, shape [#examples, 60, 130] 
    Y -- array with the whole dataset, shape [#examples,] 
    validation_folder -- scalar, number of the chosen validation folder
    test_folder -- scalar, number of the chosen test folder
    files_per_folder -- list of integers, each number stating how many audio files per folder
        
    Returns:
    The different splitted data groups
    
    """
    
    how_many = 0
    for i in range(validation_folder):
            how_many += files_per_folder[i]
            
    suma = 0
    for i in range(test_folder):
        suma += files_per_folder[i]

    X_val = X[how_many:how_many+files_per_folder[validation_folder],:,:]
    Y_val = Y[how_many:how_many+files_per_folder[validation_folder]]
    
    X_test = X[suma:suma+files_per_folder[test_folder],:,:]
    Y_test = Y[suma:suma+files_per_folder[test_folder]]
    
    if validation_folder == 9: # Handling the last case
        suma = files_per_folder[test_folder]
        X_train = X[suma:how_many,:,:]
        Y_train = Y[suma:how_many]
        
    else:  
        X_train = np.concatenate((X[0:how_many,:,:], X[how_many+files_per_folder[validation_folder]+files_per_folder[test_folder]:,:,:]) , axis = 0)
        Y_train = np.concatenate((Y[0:how_many], Y[how_many+files_per_folder[validation_folder]+files_per_folder[test_folder]:]) , axis = 0)
    

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height the spectrogram (i.e. frequency bins/bands)
    n_W0 -- scalar, width of the spectrogram (i.e. time frames)
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    #with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = 'X')
    Y = tf.placeholder(tf.float32, [None, n_y], name = 'Y')
    
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('keep_prob_conv'):
        keep_prob_conv = tf.placeholder(tf.float32)
    # Placeholder for the Batch Normalization phase
    BN_istrain = tf.placeholder(dtype = tf.bool)
    
    return X, Y, keep_prob, keep_prob_conv , BN_istrain 


def forward_propagation_with_dropout(X, keep_prob, keep_prob_conv, n_hidden_fc1, BN, BN_istrain = False):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    -> RELU -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    keep_prob -- the probability for neurons of certain layer to be kept, used for dropout regularization

    Returns:
    Z5 -- the output of the last layer
    parameters -- python dictionary containing the parameters "W1", "W2", "W3", "b1", "b2", "b3"
    
    """
    if BN:
        

        with tf.variable_scope('Conv1'):
            W1 = tf.get_variable("W", [5,5,1,24], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('b', shape = [24], initializer = tf.zeros_initializer)
            conv1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID', data_format = 'NHWC')
            conv1_bn = tf.layers.batch_normalization(conv1, axis = -1, center = True, scale = True, training = BN_istrain)
            conv1_drop = tf.nn.dropout(conv1_bn , keep_prob_conv)
            #Z1 = tf.nn.bias_add(conv1, b1)
            A1 = tf.nn.relu(conv1_drop)
            P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')
            tf.summary.histogram("weights", W1)
            tf.summary.histogram("biases", b1)
            
        with tf.variable_scope('Conv2'):
            W2 = tf.get_variable("W", [5,5,24,48], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable('b', shape = [48], initializer = tf.zeros_initializer)
            conv2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID')
            conv2_bn = tf.layers.batch_normalization(conv2, center = True, scale = True, training = BN_istrain)
            conv2_drop = tf.nn.dropout(conv2_bn , keep_prob_conv)
            #Z2 = tf.nn.bias_add(conv2, b2)
            A2 = tf.nn.relu(conv2_drop)
            P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')
            tf.summary.histogram("weights", W2)
            tf.summary.histogram("biases", b2)
            
        with tf.variable_scope('Conv3'):
            W3 = tf.get_variable("W", [5,5,48,48], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable('b', shape = [48], initializer = tf.zeros_initializer)
            conv3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='VALID')
            conv3_bn = tf.layers.batch_normalization(conv3, center = True, scale = True, training = BN_istrain)
            conv3_drop = tf.nn.dropout(conv3_bn , keep_prob_conv)
            #Z3 = tf.nn.bias_add(conv3, b3)
            A3 = tf.nn.relu(conv3_drop)
            P3 = tf.nn.max_pool(A3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')
            P3_shape = P3.get_shape().as_list()
            tf.summary.histogram("weights", W3)
            tf.summary.histogram("biases", b3)
            
        with tf.variable_scope('Flatten'):
            F = tf.reshape(P3, [-1, P3_shape[1]*P3_shape[2]*P3_shape[3]])
       
        with tf.variable_scope('FC1'):
            W4 = tf.get_variable("W", [2304,n_hidden_fc1], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable('b', shape = [n_hidden_fc1], initializer = tf.zeros_initializer)
            FC1_bn = tf.layers.batch_normalization(F, center = True, scale = True, training = BN_istrain)
            FC1_drop = tf.nn.dropout(FC1_bn, keep_prob)
            Z4 = tf.nn.relu(tf.matmul(FC1_drop , W4) + b4)
        
        with tf.variable_scope('FC2'):
            W5 = tf.get_variable("W", [n_hidden_fc1,10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.get_variable('b', shape = [10], initializer = tf.zeros_initializer)
            FC2_bn = tf.layers.batch_normalization(Z4, center = True, scale = True, training = BN_istrain)
            FC2_drop = tf.nn.dropout(FC2_bn, keep_prob)
            Z5 = tf.matmul(FC2_drop, W5) + b5
            
    else:
        with tf.variable_scope('Conv1'):
            W1 = tf.get_variable("W", [5,5,1,24], initializer=tf.contrib.layers.xavier_initializer())
            W1_drop = tf.nn.dropout(W1,keep_prob_conv)
            b1 = tf.get_variable('b', shape = [24], initializer = tf.zeros_initializer)
            conv1 = tf.nn.conv2d(X, W1_drop, strides=[1, 1, 1, 1], padding='VALID', data_format = 'NHWC')
            Z1 = tf.nn.bias_add(conv1, b1)
            A1 = tf.nn.relu(Z1)
            P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')
            tf.summary.histogram("weights", W1)
            tf.summary.histogram("biases", b1)
            tf.summary.image(name = 'Filters', tensor = tf.transpose(W1, perm = [3,0,1,2]))
     
        with tf.variable_scope('Conv2'):
            W2 = tf.get_variable("W", [5,5,24,48], initializer=tf.contrib.layers.xavier_initializer())
            W2_drop = tf.nn.dropout(W2,keep_prob_conv)
            b2 = tf.get_variable('b', shape = [48], initializer = tf.zeros_initializer)
            conv2 = tf.nn.conv2d(P1, W2_drop, strides=[1, 1, 1, 1], padding='VALID')
            Z2 = tf.nn.bias_add(conv2, b2)
            A2 = tf.nn.relu(Z2)
            P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')
            tf.summary.histogram("weights", W2)
            tf.summary.histogram("biases", b2)
            
        with tf.variable_scope('Conv3'):
            W3 = tf.get_variable("W", [5,5,48,48], initializer=tf.contrib.layers.xavier_initializer())
            W3_drop = tf.nn.dropout(W3,keep_prob_conv)
            b3 = tf.get_variable('b', shape = [48], initializer = tf.zeros_initializer)
            conv3 = tf.nn.conv2d(P2, W3_drop, strides=[1, 1, 1, 1], padding='VALID')
            Z3 = tf.nn.bias_add(conv3, b3)
            A3 = tf.nn.relu(Z3)
            P3 = tf.nn.max_pool(A3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')
            P3_shape = P3.get_shape().as_list()
            tf.summary.histogram("weights", W3)
            tf.summary.histogram("biases", b3)
    
        with tf.variable_scope('Flatten'):
                F = tf.reshape(P3, [-1, P3_shape[1]*P3_shape[2]*P3_shape[3]])

        with tf.variable_scope('FC1'):
            W4 = tf.get_variable("W", [2304,n_hidden_fc1], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable('b', shape = [n_hidden_fc1], initializer = tf.zeros_initializer)
            F1_drop = tf.nn.dropout(F, keep_prob)
            Z4 = tf.nn.relu(tf.matmul(F1_drop,W4) + b4)
 
        with tf.variable_scope('FC2'):
            W5 = tf.get_variable("W", [n_hidden_fc1,10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.get_variable('b', shape = [10], initializer = tf.zeros_initializer)
            F2_drop = tf.nn.dropout(Z4, keep_prob)
            Z5 = tf.matmul(F2_drop,W5) + b5
        

    parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, 
                  "b1": b1, "b2":b2, "b3":b3, "b4": b4, "b5": b5}


    return Z5, parameters



def compute_cost(Z5, Y):
    """
    Computes the cost
    
    Logits and labels must have the same shape, e.g. [batch_size, num_classes] 
    
    Arguments:
    Z5 -- output of forward propagation (without softmax function), of shape (number of examples,10)
    Y -- "true" labels vector placeholder, same shape as Z5
    
    Returns:
    cost - Tensor of the cost function
    """
    with tf.name_scope('cross_entropy'):
       
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=Y))
        tf.summary.scalar('cost', cost)  
    
    return cost



def compute_cost_with_regularization(Z5, Y, parameters, lambd):
    """
    Computes the cost
    
    Logits and labels must have the same shape, e.g. [batch_size, num_classes] 
    
    Arguments:
    Z5 -- output of forward propagation (without softmax function), tensor of shape (number of examples,10)
    Y -- "true" labels vector placeholder, tensor with the same shape as Z5
    parameters -- dictionary containing the values of the filters' weights
    lambd -- regularization hyperparameter
    
    Returns:
    cost - Tensor of the cost function
    """   
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    FC1_W = parameters["W4"]
    FC2_W = parameters["W5"]
    
    with tf.name_scope('Cost'):
        
        with tf.name_scope('cross_entropy'):
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=Y))
            # This gives us the cross-entropy part of the cost
            
        with tf.name_scope('L2_regularization'):
            m = tf.cast(tf.divide(tf.size(Y, out_type = tf.int32),10), dtype = tf.float32)
            # Loss function using L2 Regularization;   tf.nn.l2_loss already divides by two and squares the weights
            regularizer = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(FC1_W) + tf.nn.l2_loss(FC2_W)
            #regularizer = tf.divide(regularizer,m) # we then divide by the number of examples in the batch
            
        with tf.name_scope('total_cost'):
            cost = cross_entropy_loss + tf.multiply(lambd, regularizer)
        
        tf.summary.scalar('per_epoch_per_minibatch', cost)
          
    return cost


def batch_norm(x,
               phase,
               shift=True,
               scale=True,
               momentum=0.99,
               eps=1e-3,
               internal_update=False,
               scope=None,
               reuse=None):

    C = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [C]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        def training():
            m, v = tf.nn.moments(x, range(ndim - 1), keep_dims=True)
            update_m = _assign_moving_average(moving_m, m, momentum, 'update_mean')
            update_v = _assign_moving_average(moving_v, v, momentum, 'update_var')
            tf.add_to_collection('update_ops', update_m)
            tf.add_to_collection('update_ops', update_v)

            if internal_update:
                with tf.control_dependencies([update_m, update_v]):
                    output = (x - m) * tf.rsqrt(v + eps)
            else:
                output = (x - m) * tf.rsqrt(v + eps)
            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Get mean and variance, normalize input
        moving_m = tf.get_variable('mean', var_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_v = tf.get_variable('var', var_shape, initializer=tf.ones_initializer, trainable=False)

        if isinstance(phase, bool):
            output = training() if phase else testing()
        else:
            output = tf.cond(phase, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)

    return output



