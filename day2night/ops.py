from deepface.vgg_face import vgg_face
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import vgg16
import pdb
from rude_carnie.gender_classifer import  gender_net
weight_init = tf_contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if scope.__contains__("discriminator") :
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else :
            weight_init = tf_contrib.layers.variance_scaling_initializer()

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)

        return x + x_init

##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def adaptive_avg_pooling(x):
    # global average pooling
    gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    return gap

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

def inlrelu(x, alpha = 0.01, scope='lre_instance_norm'):
    x =  tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)
    return tf.nn.leaky_relu(x, alpha)

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

##################################################################################
# Loss function
##################################################################################

"""

Author use LSGAN
For LSGAN, multiply each of G and D by 0.5.
However, MUNIT authors did not do this.

"""

def discriminator_loss(type, real, fake):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0

    for i in range(n_scale) :
        if type == 'lsgan' :
            real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake[i]))

        if type == 'gan' :
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

        loss.append(real_loss + fake_loss)

    return sum(loss)


def generator_loss(type, fake):
    n_scale = len(fake)
    loss = []

    fake_loss = 0

    for i in range(n_scale) :
        if type == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

        if type == 'gan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

        loss.append(fake_loss)


    return sum(loss)


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss
def compute_genderface_loss(images, reuse = False):

    x_ba, Domain_B  = images[0], images[1] 
    x_ab, Domain_A  = images[2], images[3] 
    x_ba_feature = gender_net(x_ba, reuse = reuse) 
    Domain_B_feature = gender_net(Domain_B, reuse = True) 

    x_ab_feature = gender_net(x_ab, reuse = True) 
    Domain_A_feature = gender_net(Domain_A, reuse = True) 

    return tf.reduce_mean(tf.abs(x_ba_feature - Domain_B_feature)), tf.reduce_mean(tf.abs(x_ab_feature - Domain_A_feature)) 
def compute_deepface_loss(images, reuse = False):

    x_ba, Domain_B  = images[0], images[1] 
    x_ab, Domain_A  = images[2], images[3] 
    x_ba_feature = vgg_face(x_ba, reuse = reuse) 
    Domain_B_feature = vgg_face(Domain_B, reuse = True) 

    x_ab_feature = vgg_face(x_ab, reuse = True) 
    Domain_A_feature = vgg_face(Domain_A, reuse = True) 

    return tf.reduce_mean(tf.abs(x_ba_feature - Domain_B_feature)), tf.reduce_mean(tf.abs(x_ab_feature - Domain_A_feature)) 

def compute_vgg_loss( images, layer_names, batch_size, ch, vgg_weight, output_without_instance= False):
    # images = [ba, B , ab, A]
    """ Preprocess images """
    processed = [vgg_preprocess( i ) for i in images]
    #ba = vgg_preprocess( ba )
    #ab = vgg_preprocess( ab )
    #B = vgg_preprocess( B )
    #A = vgg_preprocess( A )
    
    img_shape = processed[-1].get_shape()
    h = img_shape[1]
    w = img_shape[2]
    
    layers_ = list()
    
    with tf.name_scope( "VGG16" ):
        # Create first instance of the VGG16-model
        vgg1 = vgg16.Vgg16( vgg_weight , h, w, ch )
        
        for enum, i in enumerate(processed):
            with tf.variable_scope( tf.get_variable_scope(), reuse = True ):
                with tf.name_scope( "p"+str(enum) ):
                    vgg1.build( i )
                    layers_.append(vgg1.get_layer_tensors(layer_names, enum))
        
        # vgg1.build( ba )
        # layers_ba = vgg1.get_layer_tensors( layer_names )
        # vgg1.build( A )
        # layers_A = vgg1.get_layer_tensors( layer_names )
        # vgg1.build( ab )
        # layers_ab = vgg1.get_layer_tensors( layer_names )

        vgg1.data_dict = None

        # Create second instance of the VGG16-model
        # vgg1 = vgg16.Vgg16( vgg_weight, h, w, ch )
        # vgg1.build( ba )
        # layers_ba = vgg1.get_layer_tensors( layer_names )
        
        # compute perceptual loss
        with tf.variable_scope( 'loss_a', reuse=tf.AUTO_REUSE ):
            loss_a = tf.zeros( batch_size, tf.float32 )
            loss_a_without_instance = tf.zeros( batch_size, tf.float32 )
            for f, f_ in zip( layers_[0], layers_[1] ):
                loss_a += tf.reduce_mean( tf.subtract( instance_norm(f), instance_norm(f_) ) ** 2, [ 1, 2, 3 ] )
                loss_a_without_instance += tf.reduce_mean( tf.subtract( f, f_ ) ** 2, [ 1, 2, 3 ] )
    
        with tf.variable_scope( 'loss_b', reuse=tf.AUTO_REUSE ):
            loss_b = tf.zeros( batch_size, tf.float32 )
            loss_b_without_instance = tf.zeros( batch_size, tf.float32 )
            for f, f_ in zip( layers_[2], layers_[3] ):
                loss_b += tf.reduce_mean( tf.subtract( instance_norm(f), instance_norm(f_) ) ** 2, [ 1, 2, 3 ] )
                loss_b_without_instance += tf.reduce_mean( tf.subtract( f, f_ ) ** 2, [ 1, 2, 3 ] )
        
    return (tf.reduce_mean(loss_a), tf.reduce_mean(loss_b), tf.reduce_mean(loss_a_without_instance), tf.reduce_mean(loss_b_without_instance) ) if output_without_instance else (tf.reduce_mean(loss_a), tf.reduce_mean(loss_b))

def vgg_preprocess(image, means=[123.68, 116.78, 103.94]):
    """Subtracts the given means from each image channel.
    For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    Returns:
    the centered image.
    Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
    """
    if image.get_shape().ndims < 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    print("Number of channels: %d"%num_channels)
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    
    channels = tf.split(axis=-1, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=-1, values=channels)
