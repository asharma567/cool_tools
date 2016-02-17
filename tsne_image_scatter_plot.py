from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

def image_scatter_plot(feature_M, images, img_res, res=500, cval=1., perp=30):
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

	#find the 
	max_width = images[0].shape[0]
	max_height = images[0].shape[1]
	
	tsne_scatter_plot = TSNE(init='pca',n_components=2, verbose=2, random_state=7, perplexity=perp).fit_transform(feature_M)

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
