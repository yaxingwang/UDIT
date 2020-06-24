from os import listdir
import numpy as np
import scipy.misc
import pdb
def read_image_label(data_path, _ID, batch_size=1, image_size = 128, sample_num = 20):
        Image_A_twin_set = {i: None for i in xrange(sample_num)}
        Image_A_twin_keypoint_set = {i: None for i in xrange(sample_num)}
        Noise_A_set = {i: None for i in xrange(sample_num)}

        image_path = data_path + _ID + '/train/'
        images_ = listdir(image_path)

        images = np.zeros((batch_size, image_size, image_size, 3), dtype='float32')
        labels = np.zeros((batch_size, image_size, image_size), dtype='int32')

        for n, img in enumerate(images_):

            style_a = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, 1, 1, 8])

            image = scipy.misc.imread(image_path + img)
            label = np.load(data_path + _ID  + '/train_key_point/' + img + '.npy') 
            images[n % batch_size] = (image / 127.5 - 1.)
            labels[n % batch_size] = label

            Image_A_twin_set[n] =images.copy()
            Image_A_twin_keypoint_set[n] =labels.copy()
            Noise_A_set[n] = style_a.copy()
        return Image_A_twin_set, Image_A_twin_keypoint_set, Noise_A_set
