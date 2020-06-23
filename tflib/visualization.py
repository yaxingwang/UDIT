import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.image as mping
from matplotlib import pyplot as plt
from PIL import Image
colour_code = np.array([[0, 0, 0],
                       [0, 0, 1], #BED
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]]) # WINDOW
colour_code_cityspace = np.array([[128, 64,128], # GROUND
                       [244, 35,232], #SIDEWALK
                       [70, 70, 70], #BUILDING
                       [102,102,156], #WALL
                       [190,153,153], #FENCE
                       [153,153,1531], #POLE
                       [250,170, 30], #TRAFFIC LIGHT
                       [220,220, 0], #TRAFFIC SIGN
                       [107,142, 35], #VEGETATIION
                       [152,251,152], #TERRAIN
                       [70,130,180], #SKY
                       [220, 20, 60], #PERSION
                       [255, 0, 0], #RIDER
                       [0, 0,142],#car
		       [0, 0, 70],#TRUCK
		       [0, 60,100],#BUS
		       [0, 80,100], #TRAIN
		       [0, 0,230],#MOTORCYCLE
		       [119, 11, 32],#BICYCLE
		       [0, 0, 0]])/255. #unknow
                                                

def _discrete_matshow_adaptive(data, labels_names=[], counter = 0, title=""):
    """Displays segmentation results using colormap that is adapted
    to a number of classes. Uses labels_names to write class names
    aside the color label. Used as a helper function for 
    visualize_segmentation_adaptive() function.
    
    Parameters
    ----------
    data : 2d numpy array (width, height)
        Array with integers representing class predictions
    labels_names : list
        List with class_names
    """
    
    fig_size = [14, 12]
    plt.rcParams["figure.figsize"] = fig_size
    
    #get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)
    
    # set limits .5 outside true range
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)
    
    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))
    
    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    
    #if title:
    #    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.savefig('/home/yaxing/image_to_depth_label_max_medain_l2/visualization/annotation_%d.png'%counter, bbox_inches='tight', pad_inches = 0)
    plt.gcf().clear()
    #plt.show()

        
def _visualize_segmentation_adaptive(predictions, segmentation_class_lut, counter= 0, title="Segmentation"):
    """Displays segmentation results using colormap that is adapted
    to a number of classes currently present in an image, instead
    of PASCAL VOC colormap where 21 values is used for
    all images. Adds colorbar with printed names against each color.
    Number of classes is renumerated starting from 0, depending
    on number of classes that are present in the image.
    
    Parameters
    ----------
    predictions : 2d numpy array (width, height)
        Array with integers representing class predictions
    segmentation_class_lut : dict
        A dict that maps class number to its name like
        {0: 'background', 100: 'airplane'}
        
    """
    
    # TODO: add non-adaptive visualization function, where the colorbar
    # will be constant with names
    

    unique_classes, relabeled_image = np.unique(predictions,
                                                return_inverse=True)

    relabeled_image = relabeled_image.reshape(predictions.shape)

    labels_names = []

    for index, current_class_number in enumerate(unique_classes):

        labels_names.append(str(index) + ' ' + segmentation_class_lut[current_class_number])

    _discrete_matshow_adaptive(data=relabeled_image, labels_names=labels_names, counter = counter, title=title)


def visualize_depth_image_synth_images(raw_depth,predict_depth, counter= 0, title="Segmentation"):

    fig = plt.figure()
    # raw depth
    a = fig.add_subplot(1,2,1)
    anno_pred = plt.imshow(raw_depth)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Raw_depth')
    # prediction depth
    a = fig.add_subplot(1,2,2)
    anno_pred = plt.imshow(predict_depth)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Pre_depth')
    
    fig.savefig('./visualization/depth/photo_annotation_depth_%d.png'%counter , bbox_inches='tight', pad_inches = 0, dpi=100)
    plt.gcf().clear()
 #   def visualize_depth():
 #   import matplotlib.pyplot as plt
 #   import numpy as np
 #
 #   grid = np.random.random((10,10))
 #
 #   fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))
 #
 #   ax1.imshow(grid)
 #   ax1.set_title('Default')
 #
 #   ax2.imshow(grid)
 #   ax2.set_title('Auto-scaled Aspect')
 #
 #   ax3.imshow(grid)
 #   ax3.set_title('Manually Set Aspect')
 #
 #   plt.tight_layout()
 #   plt.show()

def save_class_from_instance(predictions,counter, visualization_path = 'visualization'):

    h,w  = predictions.shape

    class_img_rgb_raw = np.zeros((h,w,3),dtype=np.uint8)
    r_raw = class_img_rgb_raw[:,:,0]
    g_raw = class_img_rgb_raw[:,:,1]
    b_raw = class_img_rgb_raw[:,:,2]
    predictions[predictions==255] =0

    for i in xrange(h):
	    for j in xrange(w):
 	       r_raw[i,j] = np.uint8(colour_code_cityspace[predictions[i,j]][0]*255)
 	       g_raw[i,j] = np.uint8(colour_code_cityspace[predictions[i,j]][1]*255)
 	       b_raw[i,j] = np.uint8(colour_code_cityspace[predictions[i,j]][2]*255)

    class_img_rgb_raw[:,:,0] = r_raw
    class_img_rgb_raw[:,:,1] = g_raw
    class_img_rgb_raw[:,:,2] = b_raw
    class_img_rgb_raw = Image.fromarray(class_img_rgb_raw)
    class_img_rgb_raw.save('%s/annotation_%d.png'%(visualization_path,counter))
def vis_gt(source_train_image_gt, source_train_label_gt, target_val_image_gt, target_val_label_gt,visualization_path = None, counter = 0):
    from scipy.misc import imsave
    imsave('%s/source_train_image_gt.png'%visualization_path, source_train_image_gt)
    imsave('%s/target_val_image_gt.png'%visualization_path, target_val_image_gt)
    save_class_from_instance(source_train_label_gt,  counter= counter, visualization_path = visualization_path )
    save_class_from_instance(target_val_label_gt,  counter= counter + 1, visualization_path = visualization_path)
  
    fig_size = [18, 15]
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()
    a = fig.add_subplot(2,2,1)
    raw_image = mping.imread('%s/source_train_image_gt.png'%visualization_path)
    im = plt.imshow(raw_image)##gray,hot
    plt.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    a.set_title('source_image')


    a = fig.add_subplot(2,2,2)
    annotation_prediction = mping.imread('%s/annotation_%d.png'%(visualization_path, counter))
    anno_gr = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_gr.axes.get_xaxis().set_visible(False)
    anno_gr.axes.get_yaxis().set_visible(False)
    a.set_title('source_label')

    a = fig.add_subplot(2,2,3)
    raw_image = mping.imread('%s/target_val_image_gt.png'%visualization_path)
    im = plt.imshow(raw_image)##gray,hot
    plt.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    a.set_title('target_image')

    a = fig.add_subplot(2,2,4)
    annotation_prediction = mping.imread('%s/annotation_%d.png'%(visualization_path, counter + 1))
    anno_gr = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_gr.axes.get_xaxis().set_visible(False)
    anno_gr.axes.get_yaxis().set_visible(False)
    a.set_title('target_label')
    fig.savefig('%s/image_label_%d.png'%(visualization_path,counter), pad_inches = 0, dpi=100)
    plt.gcf().clear()



def visualize_image_annotation(raw_image, raw_annotation, pred_map, class_labels,  counter= 0, visualization_path = 'visualization'):
    # visualize the raw annotation  
    #_visualize_segmentation_adaptive(predictions[0], segmentation_class_lut, counter= counter, title="raw annotation")
    #_visualize_segmentation_adaptive(predictions[1], segmentation_class_lut, counter= counter + 1, title="predict annotation")
    save_class_from_instance(raw_annotation,  counter= counter, visualization_path = visualization_path )
    save_class_from_instance(pred_map,  counter= counter + 1, visualization_path = visualization_path)
  
    fig_size = [14, 12]
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()
    a = fig.add_subplot(3,1,1)
    raw_image = mping.imread('%s/raw_image.png'%visualization_path)
    im = plt.imshow(raw_image)##gray,hot
    plt.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    a.set_title('GT_image')


    a = fig.add_subplot(3,1,2)
    annotation_prediction = mping.imread('%s/annotation_%d.png'%(visualization_path,counter))
    anno_gr = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_gr.axes.get_xaxis().set_visible(False)
    anno_gr.axes.get_yaxis().set_visible(False)
    a.set_title('GT_label')

    a = fig.add_subplot(3,1,3)
    annotation_prediction = mping.imread('%s/annotation_%d.png'%(visualization_path,counter+1))
    anno_gr = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_gr.axes.get_xaxis().set_visible(False)
    anno_gr.axes.get_yaxis().set_visible(False)
    a.set_title('Predicted_label')



    fig.savefig('%s/annotation_depth_%d.png'%(visualization_path,counter), bbox_inches='tight', pad_inches = 0, dpi=100)
    plt.gcf().clear()


def visualize_segmentation_depth(_GT_segmentation, _Parent_segmentation, _Child_segmentation, _GT_depth, _Parent_depth, _Child_depth, segmentation_class_lut, counter= 0, title="Segmentation",GT=None,visualization_path = 'visualization'):
    # visualize the raw annotation  
    #_visualize_segmentation_adaptive(predictions[0], segmentation_class_lut, counter= counter, title="raw annotation")
    #_visualize_segmentation_adaptive(predictions[1], segmentation_class_lut, counter= counter + 1, title="predict annotation")
    save_class_from_instance(_GT_segmentation,  counter= counter, visualization_path = visualization_path )
    save_class_from_instance(_Parent_segmentation,  counter= counter + 1, visualization_path = visualization_path)
    save_class_from_instance(_Child_segmentation,  counter= counter + 2, visualization_path = visualization_path)
  
    fig_size = [14, 12]
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()
    a = fig.add_subplot(3,2,1)
    # raw depth
    anno_pred = plt.imshow(_GT_depth)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('GT_depth')


    a = fig.add_subplot(3,2,2)
    annotation_prediction = mping.imread('./%s/annotation_%d.png'%(visualization_path,counter))
    anno_gr = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_gr.axes.get_xaxis().set_visible(False)
    anno_gr.axes.get_yaxis().set_visible(False)
    a.set_title('GT_label')


    a = fig.add_subplot(3,2,3)
    anno_pred = plt.imshow(_Parent_depth)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Parent_depth')

    a = fig.add_subplot(3,2,4)
    annotation_prediction = mping.imread('./%s/annotation_%d.png'%(visualization_path,counter + 1))
    anno_pred = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Parent_label')


    a = fig.add_subplot(3,2,5)
    anno_pred = plt.imshow(_Child_depth)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Child_depth')

    a = fig.add_subplot(3,2,6)
    annotation_prediction = mping.imread('./%s/annotation_%d.png'%(visualization_path,counter + 2))
    anno_pred = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Child_label')


    fig.savefig('./%s/annotation_depth_%d.png'%(visualization_path,counter), bbox_inches='tight', pad_inches = 0, dpi=100)
    plt.gcf().clear()

def visualize_raw_image_synth_images(raw_img,raw_depth,predict_depth,generated_image,raw_img2,generated_image2, predictions, segmentation_class_lut, counter= 0, title="Segmentation",GT=None,visualization_path = 'visualization'):
    # visualize the raw annotation  
    #_visualize_segmentation_adaptive(predictions[0], segmentation_class_lut, counter= counter, title="raw annotation")
    #_visualize_segmentation_adaptive(predictions[1], segmentation_class_lut, counter= counter + 1, title="predict annotation")
    save_class_from_instance(predictions[0],  counter= counter, visualization_path = visualization_path )
    save_class_from_instance(predictions[1],  counter= counter + 1, visualization_path = visualization_path)
    save_class_from_instance(predictions[2],  counter= counter + 2, visualization_path = visualization_path)
    save_class_from_instance(predictions[3],  counter= counter + 3, visualization_path = visualization_path)
  
    fig_size = [14, 12]
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()
    a = fig.add_subplot(5,2,1)
    raw_image = mping.imread('./%s/raw_image.png'%visualization_path)
    im = plt.imshow(raw_image)##gray,hot
    plt.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    #fig.savefig('proba_global.png', bbox_inches='tight', pad_inches = 0, dpi=100)
    a.set_title('Raw_Photo')

    a = fig.add_subplot(5,2,2)
    pred_image = mping.imread('./%s/pre_image.png'%visualization_path)
    im = plt.imshow(pred_image)##gray,hot
    plt.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    #fig.savefig('proba_global.png', bbox_inches='tight', pad_inches = 0, dpi=100)
    a.set_title('Gen_Photo')


    a = fig.add_subplot(5,2,3)
    annotation_prediction = mping.imread('./%s/annotation_%d.png'%(visualization_path,counter))
    anno_gr = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_gr.axes.get_xaxis().set_visible(False)
    anno_gr.axes.get_yaxis().set_visible(False)
    a.set_title('Raw_label')

    a = fig.add_subplot(5,2,4)
    annotation_prediction = mping.imread('./%s/annotation_%d.png'%(visualization_path,counter + 2))
    anno_pred = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Pre_label')



    a = fig.add_subplot(5,2,5)
    raw_image = mping.imread('./%s/raw_image2.png'%visualization_path)
    im = plt.imshow(raw_image)##gray,hot
    plt.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    #fig.savefig('proba_global.png', bbox_inches='tight', pad_inches = 0, dpi=100)
    a.set_title('Raw_Photo')

    a = fig.add_subplot(5,2,6)
    pred_image = mping.imread('./%s/pre_image2.png'%visualization_path)
    im = plt.imshow(pred_image)##gray,hot
    plt.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    #fig.savefig('proba_global.png', bbox_inches='tight', pad_inches = 0, dpi=100)
    a.set_title('Gen_Photo')



    # raw depth
    a = fig.add_subplot(5,2,7)
    anno_pred = plt.imshow(raw_depth)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Raw_depth')

    # prediction depth
    
    a = fig.add_subplot(5,2,8)
    anno_pred = plt.imshow(predict_depth)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Pred_depth')

    # raw depth
    a = fig.add_subplot(5,2,9)
    annotation_prediction = mping.imread('./%s/annotation_%d.png'%(visualization_path,counter + 3))
    anno_pred = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Raw_label_depth')

    # test from depth to label
    a = fig.add_subplot(5,2,10)
    annotation_prediction = mping.imread('./%s/annotation_%d.png'%(visualization_path,counter + 1))
    anno_pred = plt.imshow(annotation_prediction)
    plt.axis('off')
    anno_pred.axes.get_xaxis().set_visible(False)
    anno_pred.axes.get_yaxis().set_visible(False)
    a.set_title('Pre_label_depth')



    fig.savefig('./%s/photo_annotation_depth_%d.png'%(visualization_path,counter), bbox_inches='tight', pad_inches = 0, dpi=100)
    plt.gcf().clear()

