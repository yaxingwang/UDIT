from os import listdir
import numpy as np
import scipy.misc
import time
import pdb

def make_generator(path, n_files, batch_size,image_size):
    epoch_count = [1]
    image_list = listdir(path)
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        if n_files > len(image_list):
            #print('No Enough Images!!!!!')
            files = range(len(image_list))
        else:
            files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1        
        for n, i in enumerate(files):
            #image = scipy.misc.imread("{}/{}.png".format(path, str(i+1).zfill(len(str(n_files)))))            
            image = scipy.misc.imread("{}".format(path + image_list[i]))

	    image = scipy.misc.imresize(image,(image_size,image_size))
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir='/home/ishaan/data/imagenet64',image_size = 64, NUM_TRAIN = 7000):
    return (
        make_generator(data_dir+'/train/', NUM_TRAIN, batch_size,image_size),
        make_generator(data_dir+'/val/', 5000, batch_size,image_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
