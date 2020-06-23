import tensorflow as tf
import numpy as np
import pdb

average = np.array([129.18628, 104.76238,  93.59396], dtype=np.float32)
layers =  np.load('./deepface/layers.npy')

def vgg_face(input_maps, reuse = False):
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=input_maps)
    bgr = tf.concat(axis=3, values=[
        red - average[0],
        green - average[1],
        blue- average[2],
    ])
    # read layer info
    current = input_maps
    network = {}
    Name = None
    with tf.variable_scope('deepface',reuse = reuse ) as scope:
        for layer in layers[0]:
            name = layer[0]['name'][0][0]
            layer_type = layer[0]['type'][0][0]
            if Name != 'conv5_3':
                if layer_type == 'conv':
                    if name[:2] == 'fc':
                        padding = 'VALID'
                    else:
                        padding = 'SAME'
                    stride = layer[0]['stride'][0][0]
                    kernel, bias = layer[0]['weights'][0][0]
                    # kernel = np.transpose(kernel, (1, 0, 2, 3))
                    bias = np.squeeze(bias).reshape(-1)
                    conv = tf.nn.conv2d(current, tf.get_variable(name,initializer=kernel),
                                        strides=(1, stride[0], stride[0], 1), padding=padding)
                    current = tf.nn.bias_add(conv, bias)
                    print name, 'stride:', stride, 'kernel size:', np.shape(kernel)

                elif layer_type == 'relu':
                    current = tf.nn.relu(current)
                    print name
                elif layer_type == 'pool':
                    stride = layer[0]['stride'][0][0]
                    pool = layer[0]['pool'][0][0]
                    current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                             strides=(1, stride[0], stride[0], 1), padding='SAME')
                    print name, 'stride:', stride
                elif layer_type == 'softmax':
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
                    print name

                network[name] = current
                Name = name.copy()

    return network['conv5_3']

