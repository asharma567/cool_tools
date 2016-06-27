from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from cool_decorators import multithread_map
from PIL import Image
import os, glob



def image_scatter_plot(feature_M, images, res=500, cval=1., perp=30, tsne_initialization='pca'):
    '''
    I:
        feature_M: numpy array
        Features to visualize

        images: list or numpy array
            Corresponding images to feature_M. Expects float images from (0,1).

        img_res: float or int
            Resolution to embed images at

        res: float or int
            Size of embedding image in pixels

        cval: float or numpy array
            Background color value

    O: 
        canvas: numpy array
    '''

    #find the max ratios
    max_width = images[0].shape[0]
    max_height = images[0].shape[1]
    
    tsne_scatter_plot = TSNE(init=tsne_initialization,n_components=2, verbose=2, perplexity=perp).fit_transform(feature_M)

    xx = tsne_scatter_plot[:, 0]
    yy = tsne_scatter_plot[:, 1]
    
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    
    # Fix the ratios
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = sx / float(sy) * res
        res_y = res
    else:
        res_x = res
        res_y = sy / float(sx) * res

    #create a blank canvas
    canvas = np.ones((res_x + max_width, res_y + max_height, 3)) * cval
    
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    
    #fill coordinates with the images
    for x, y, image in zip(xx, yy, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        canvas[x_idx:x_idx + w, y_idx:y_idx + h] = image

    return canvas

def main(path, save_fname):
    
    def rescaler(im_file):
        try:
            im = Image.open(im_file)
            im = im.resize(size, Image.ANTIALIAS)
            return im

        except IOError:
            print "cannot create thumbnail for '%s'" % im_file
            return None
    
    print '=' * 100
    print 'Looking in the following path--'
    print path
    print '=' * 100

    #rescales
    arrays_rescaled = multithread_map(rescaler, glob.glob(path))
    X = map(lambda x: np.array(np.array(x, dtype=np.float32)), arrays_rescaled)
    pics_feature_matrix = rgb_arrays_to_flattened_feature_matrix(X)
    
    #parameters
    input_res = 1000
    perplexity = 30
    size = 32
    
    plot_data = image_scatter_plot(pics_feature_matrix, X, X[0].shape[0], res=input_res, cval=1., perp=perplexity)

    print(plot_data.shape)
    im = Image.fromarray(np.uint8(plot_data))
    im.show()
    if save_fname:
        im.save(save_fname, 'JPEG')



def rgb_arrays_to_flattened_feature_matrix(array_of_images):
    '''
    I: a list of numpy arrays of images eg [np.array(image_file1), np.array(image_file2)]
    O: numpy array of these images flattened into row form
    '''

    pics = np.array(array_of_images[0].ravel())
    for index in range(1, len(array_of_images)):
        pics = np.c_[pics, array_of_images[index].ravel()]
    return pics.T


if __name__ == '__main__':
    main(
        path='/Users/asharma/Desktop/computer_vision/test_set/0/*.jpeg', 
        save_fname='testing.jpeg'
        )

