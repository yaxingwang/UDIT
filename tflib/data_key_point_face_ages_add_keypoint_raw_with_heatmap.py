
from os import listdir
import numpy as np
import scipy.misc
import time
import copy
import pdb
from visualization import vis_gt 
from PIL import Image
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
def heatmap(label, image_size,size_heatmap = 16):
    images = np.zeros((image_size, image_size, 68), dtype='float32')
    for i in xrange(68):
        itemindex = np.where(label == i)
        try:
            r_, c_ = (itemindex[0][0], itemindex[1][0])
            r_begin = int((r_ - size_heatmap/2) > 0)*(r_ - size_heatmap/2)
            r_end = int((image_size - r_ - size_heatmap/2) > 0) * (r_ + size_heatmap/2) + (1 -int((image_size - r_ - size_heatmap/2) > 0))*image_size  

            c_begin = int((c_ - size_heatmap/2) > 0)*(c_ - size_heatmap/2) 
            c_end = int((image_size - c_ - size_heatmap/2) > 0) * (c_ + size_heatmap/2) + (1 -int((image_size - c_ - size_heatmap/2) > 0))*image_size  

            heatmap_ = np.load('/datatmp/code/MUNIT-Tensorflow-used_vgg_loss/dataset/face2face_key_point_ages_add_label_raw/heatmap/%d_%d.npy'%(np.random.random_integers(0, image_size, 1)[0],np.random.random_integers(0, image_size, 1)[0]))
            images[r_begin:r_end, c_begin:c_end, i] = heatmap_[:(r_end-r_begin ), :(c_end- c_begin)]
        except:
            pass
    return images

def transform_label(label_orig, sz):
    """
    Function to transform the predictions obtained to original size of cityscapes labels
    """
    label = copy.deepcopy(label_orig)
    label = Image.fromarray(label.squeeze().astype(np.uint8))
    label = label.resize( (sz[0],sz[1]),Image.NEAREST)
    label = np.array(label, dtype=np.int32)
    return label
def transform_deep(deep_orig, sz):
    """
    Function to transform the predictions obtained to original size of cityscapes labels
    """
    deep = copy.deepcopy(deep_orig)
    deep = Image.fromarray(deep.squeeze().astype(np.float32))
    deep = deep.resize( (sz[0],sz[1]),Image.NEAREST)
    deep = np.array(deep, dtype=np.float32)
    return deep 

def make_generator(path, batch_size, image_size, pharse='train', domain = None, start_point = 1):
    epoch_count = [start_point]
    if domain =='domainA':
        if pharse == 'train':
            file_image = 'trainA'
            file_key_point = 'trainA_key_point'
            file_key_point_raw = 'trainA_label'
        else:
            file_image = 'testA'
            file_key_point = 'testA_key_point'
            file_key_point_raw = 'testA_label'
    if domain =='domainB':
        if pharse == 'train':
            file_image = 'trainB'
            file_key_point = 'trainB_key_point'
            file_key_point_raw = 'trainB_label'
        else:
            file_image = 'testB'
            file_key_point = 'testB_key_point'
            file_key_point_raw = 'testB_label'
    images_index = listdir(path + file_image)


    def get_epoch():
        images = np.zeros((batch_size, image_size, image_size, 3), dtype='float32')
        images_twin = np.zeros((batch_size, image_size, image_size, 3), dtype='float32')
        labels = np.zeros((batch_size, image_size, image_size), dtype='int32')
        labels_raw = np.zeros((batch_size, 136), dtype='float32')
        labels_twin = np.zeros((batch_size, image_size, image_size), dtype='int32')
        labels_twin_raw = np.zeros((batch_size, 136), dtype='float32')
        labels_heatmap = np.zeros((batch_size, image_size, image_size, 68), dtype='float32')
        labels_twin_heatmap = np.zeros((batch_size, image_size, image_size, 68), dtype='float32')

        files = range(len(images_index))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = scipy.misc.imread("%s"%(path + file_image + '/' + images_index[i]))
            image_twin = scipy.misc.imread("%s"%(path + file_image + '/' + images_index[i-epoch_count[0]]))

            label = np.load(path + file_key_point +  '/'+images_index[i] + '.npy') 
            label_raw = np.load(path + file_key_point_raw +  '/'+images_index[i] + '.npy').astype(np.float)
            label_twin = np.load(path + file_key_point +  '/'+images_index[i-epoch_count[0]] + '.npy') 
            label_twin_raw = np.load(path + file_key_point_raw +  '/'+images_index[i-epoch_count[0]] + '.npy').astype(np.float) 
            label_heatmap = heatmap(label, image_size)
            label_twin_heatmap = heatmap(label_twin, image_size)
             
	    #label = scipy.misc.imresize(label,(image_size/2,image_size), interp = 'nearest')
            images[n % batch_size] = (image / 127.5 - 1.)
            images_twin[n % batch_size] = (image_twin / 127.5 - 1.)
            labels[n % batch_size] = label
            labels_twin[n % batch_size] = label_twin

            label_raw[label_raw < 0] = 0.
            labels_raw[n % batch_size] = label_raw/image_size
            
            label_twin_raw[label_twin_raw < 0] = 0.
            labels_twin_raw[n % batch_size] = label_twin_raw/image_size

            labels_heatmap[n % batch_size] = label_heatmap.copy()*2. -1.
            labels_twin_heatmap[n % batch_size] = label_twin_heatmap.copy() *2. -1.

            if n > 0 and n % batch_size == 0:
                yield (images,labels,labels_raw,images_twin, labels_twin,labels_twin_raw, labels_heatmap, labels_twin_heatmap)
    return get_epoch

def load(batch_size, data_dir='face2cat',image_size = 1024, domain = None, data_path = 'dataset/', start_point = 1):
    return make_generator(data_path+data_dir+'/', batch_size, image_size, pharse='train', domain = domain, start_point = start_point),\
           make_generator(data_path+data_dir+'/', batch_size, image_size, pharse='train', domain = domain, start_point = start_point)
           
    

if __name__ == '__main__':
    pdb.set_trace()
    train_gen, test_gen = load(2, data_dir='face2cat',image_size = 128, domain = 'domainA')
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        _image = batch[0][1]
        _label = batch[1][1]
        vis_gt(_image, _label, _image, _label, './')
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
