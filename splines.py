import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import math
import pandas as pd
import scipy.io
from scipy import interpolate

keys = ['names', 'images', 'drawings', 'offsets']

def load_mat(file):
	mat = scipy.io.loadmat(file)
	return mat
def mat_to_dict(filename="./data_background.mat"):
    mat_dict = load_mat(filename)

    names = []
    for i in range(len(mat_dict['names'])):
        names.append(str(mat_dict['names'][i].tolist()[0][0]))

    character = {}
    for index, (name, drawings_dir, images_dir) in enumerate(zip(names, mat_dict['drawings'], mat_dict['images'])):

        drawings_dir, images_dir = drawings_dir[0], images_dir[0]

        _dir = []
        for drawings_subdir, images_subdir in zip(drawings_dir, images_dir):
            drawings_subdir, images_subdir = drawings_subdir[0], images_subdir[0]
            _subdir = []
            for drawings_img, images_img in zip(drawings_subdir, images_subdir):
                idrawings_img, images_img = drawings_img[0], images_img[0]

                _img = {}
                _img['data'] = images_img
                _img['strokes'] = []
                for drawings_stroke in zip(drawings_img):
                    _img['strokes'].append(drawings_stroke[0])
                _subdir.append(_img)
            _dir.append(_subdir)

        character[name] = _dir
    return character

def dict_images(filename="./data_background.mat"):
    #  images[lang][character][0]['image'][instance][0] -> (105,105) numpy ndarray
    mat_dict=load_mat(filename)
    images={}
    names = []

    for i in range(len(mat_dict['names'])):
        names.append(str(mat_dict['names'][i].tolist()[0][0]))

    for index,(name,images_dir) in enumerate(zip(names,mat_dict['images'])):
        images_dir=images_dir[0]
        
        _dir=[]
        for images_subdir in zip(images_dir):
            images_subdir = images_subdir[0]
            _subdir = []
            for images_img in zip(images_subdir):
                images_img = images_img[0]
                _img = {}
                _img['image'] = images_img
                _subdir.append(_img)
            _dir.append(_subdir)
        images[name] = _dir

    return images


def rep(arr):
    last=arr[-1].reshape(-1,1)
    for _ in range(25-arr.shape[0]):
        arr=np.append(arr,last,axis=0)
    return arr

def spline_5(primitive,k):
    """
    Args:
        primitive (numpy array): The (n,2) np array of the primitive coordinates.
        k (int): Length of spline
    """
    xc=primitive[:,[0]]
    yc=primitive[:,[1]]
    d=xc.shape[0]
    try :
        if d>=5:
            tck, u = interpolate.splprep([xc.reshape(-1), yc.reshape(-1)], s=0)
            unew = np.arange(0,1+1/k,1/(k-1))
            out = interpolate.splev(unew, tck)

# #             # Comment out below 4 lines for outputting the spline
#             xn, yn = out[0], out[1]
#             tck, u = interpolate.splprep([xn, yn], s=0)
#             unew2 = np.arange(0,1,0.05)
#             out = interpolate.splev(unew2, tck)

            return np.array(out).transpose((1,0))
        else :
            if d==1:
                return np.concatenate((rep(xc),rep(yc)),axis=1)
            tck, u = interpolate.splprep([xc.reshape(-1), yc.reshape(-1)], s=0,k=1)
            unew = np.arange(0,1+1/k,1/(k-1))
            out = interpolate.splev(unew, tck)
            return np.array(out).transpose((1,0))
    except :
        xc+=np.random.normal(0,0.5,(d,1))
        yc+=np.random.normal(0,0.5,(d,1))
        return spline_5(np.concatenate((xc,yc),axis=1),k)

def spline_dict(character):
    spline_prims = {}
    spl_len=25
    for lang in character :
        # lang are keys
        spline_prims[lang]=[]
        # spline_prims[lang][char][inst][0][primitive]
        for i,char in enumerate(character[lang]) :
            # char = characters of lang
            spline_prims[lang].append([])
            for j,inst in enumerate(char) :
                # inst = character instance
                spline_prims[lang][i].append([[]])
                for k,primitive in enumerate(inst['strokes'][0]):
                    # primitive = single element list which contains primitive numpy array
    #                 print(primitive[0].shape)
    #                 print('iteration ',k,j)
                    spline_prims[lang][i][j][0].append(spline_5(primitive[0],spl_len))
    return spline_prims

def plot_char(char,typ=''):
    for _,c in enumerate(char):
        plt.plot(c[:,0],c[:,1],typ)
    plt.axis('square')
    return plt.show()

