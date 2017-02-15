"""
(CONV -> RELU -> POOL) * 2 -> FC -> RELU -> FC
After 600 (on avg 3it/s) 0.917619 on 0.2 validation

((CONV -> RELU)*2 -> POOL) * 2  -> (FC -> RELU) * 2 -> FC
After 600 (on avg 1.3it/s) 0.944405 (without dropout 0.923333) on 0.2 validation
After 2600 (on avg 1.37it/s) 0.953214 on 0.2 validation
After 5000 (on avg 1.26) 0.96381  on 0.2 validation


28 x 28 x 1        3x3x16
26 x 26 x 16
"""
from tflearn import conv_2d, max_pool_2d, fully_connected, dropout
import tensorflow as tf





# TODO add parameter to the network to generalize it
def network(input_layer, drop_out=0.5):
    # net = tf.reshape(input_layer, [-1, 28, 28, 1], name='reshape')
    net = conv_2d(net, nb_filter=16, filter_size=[3, 3], activation='relu', name='convolutional layer 1')
    net = conv_2d(net, nb_filter=32, filter_size=[3, 3], activation='relu', name='convolutional layer 2')
    net = max_pool_2d(net, kernel_size=2, name='pool layer 1')
    net = conv_2d(net, nb_filter=64, filter_size=[3, 3], activation='relu', name='convolutional layer 3')
    net = conv_2d(net, nb_filter=128, filter_size=[3, 3], activation='relu', name='convolutional layer 4')
    net = max_pool_2d(net, kernel_size=2, name='pool layer 2')
    net = fully_connected(net, n_units=2048, activation='relu', name='fully connected 1')
    net = fully_connected(net, n_units=512, activation='relu', name='fully connected 2')
    net = dropout(net, 1 - drop_out, name='dropout')
    net = fully_connected(net, n_units=10, activation='linear', name='fully connected 3')
    return net


