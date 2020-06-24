from os import listdir
import numpy as np
import scipy.misc
import time
import pdb
from visualization import vis_gt 
Label={'bedroom':0,
       'kitchen':1,
       'dining_room':2,
       'conference_room':3,
       'living_room':4,
       'bridge':5,
       'tower':6,
       'classroom':7,
       'church_outdoor':8,
       'restaurant':9}

def make_generator(path, batch_size, image_size, pharse='train'):
    epoch_count = [1]
    images_path = [line.strip() for  line in open('./cityscapes_dataname/%s.txt'%pharse, 'r')] 
    images_list = []
    labels_list = []
    for image_path  in images_path:
        images_list.extend([path + '/leftImg8bit/%s/'%pharse+ image_path])
    if pharse == 'val':
        labels_path = [line.strip() for  line in open('./cityscapes_dataname/label.txt', 'r')] 
        for label_path in labels_path:
            labels_list.extend([path + '/gtFine/val/' + label_path])

    def get_epoch():
        images = np.zeros((batch_size, image_size/2, image_size, 3), dtype='float32')
        labels = np.zeros((batch_size, image_size/2, image_size), dtype='int32')
        images_name = {i: None for i in xrange(batch_size)} 
        files = range(len(images_list))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = scipy.misc.imread("{}".format(images_list[i]))
	    image = scipy.misc.imresize(image,(image_size/2, image_size))
            images[n % batch_size] = image/127.5 -1
            images_name[n % batch_size] = images_list[i].split('/')[-1]
            if pharse == 'train':
                if n > 0 and n % batch_size == 0:
                    yield (images, images_name)

            else:
                label = scipy.misc.imread("{}".format(labels_list[i]))
                label = scipy.misc.imresize(label,(image_size/2,image_size), interp = 'nearest' )
                labels[n % batch_size] = label
                if n > 0 and n % batch_size == 0:
                    yield (images, labels, images_name)
    return get_epoch

def load(batch_size, data_dir='/home/yaxing/data/leftImg8bit_trainvaltest',image_size = 1024):
    return (
        make_generator(data_dir , batch_size, image_size, pharse='train'),
        make_generator(data_dir , batch_size, image_size, pharse='val')
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(2, data_dir='/home/yaxing/data/leftImg8bit_trainvaltest',image_size = 1024)
    t0 = time.time()
    pdb.set_trace()
    for i, batch in enumerate(valid_gen(), start=1):
        _image = batch[0][0].transpose(1, 2, 0)
        _label = batch[1][0]
        vis_gt(_image, _label, _image, _label, './')
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
