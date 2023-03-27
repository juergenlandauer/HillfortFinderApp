#!/usr/bin/env python
# coding: utf-8

import numpy as np

def normalizeImg(img, NORMALIZER):
    ''' takes an image and applies a normalizer function to it
        - images could be a Pillow image or a Numpy array 
    '''
    
    # min/max excluding NaN values
    tileMin = np.min(np.ma.masked_array(img, np.isnan(img)))
    tileMax = np.max(np.ma.masked_array(img, np.isnan(img)))
    
    # normalizations from 
    # https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
    if NORMALIZER == "NONE": # do nothing
        pixels = img        
 
    elif NORMALIZER == "MIN":# deduct minimum
        pixels = img - tileMin
    
    elif NORMALIZER == "SCALE":# scale between 0..255
        #pixels = img / 255.0
        pixels = np.interp(img, (tileMin, tileMax), (0, 255))#.astype(np.uint8)
    
    elif NORMALIZER == "GlobCenter":
        # possibly convert from integers to floats
        pixels = pixels.astype('float32')
        # calculate global mean
        mean = pixels.mean()
        print('Mean: %.3f' % mean)
        print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
        # global centering of pixels
        pixels = pixels - mean
        # confirm it had the desired effect
        #mean = pixels.mean()
        print('Mean: %.3f' % mean)
        print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

    elif NORMALIZER == "LocCenter":
        # possibly convert from integers to floats
        pixels = pixels.astype('float32')
        means = pixels.mean(axis=(0,1), dtype='float64') # calculate per-channel means and standard deviations
        print('Means: %s' % means)
        print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
        # per-channel centering of pixels
        pixels -= means
        # confirm it had the desired effect
        means = pixels.mean(axis=(0,1), dtype='float64')
        print('Means: %s' % means)
        print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))

    elif NORMALIZER == "PosGlobStd":
        # convert from integers to floats
        pixels = pixels.astype('float32')
        # calculate global mean and standard deviation
        mean, std = pixels.mean(), pixels.std()
        #print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
        # global standardization of pixels
        pixels = (pixels - mean) / std
        # clip pixel values to [-1,1]
        pixels = np.clip(pixels, -1.0, 1.0)
        # shift from [-1,1] to [0,1] with 0.5 mean
        pixels = ((pixels + 1.0) * 255./2).astype('uint8')
        # confirm it had the desired effect
        mean, std = pixels.mean(), pixels.std()
        print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
        print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

    else:
        print ("unknown normalizer:", NORMALIZER)
        pixels = None
    
    return pixels

