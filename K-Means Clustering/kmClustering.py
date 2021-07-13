# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 11:29:06 2021

@author: Sanjay's
"""

import numpy as np
from skimage import io, img_as_float
import matplotlib.pyplot as plt

def clustering(image_vectors, k):
    num_of_itr = 3
    pixel_to_mean = np.full((image_vectors.shape[0],), -1)
    #mean is the random k means with 3 points representing a pixel 
    mean = np.random.rand(k, 3)

    for i in range(num_of_itr):
        
        #pixel_in_cluster holds each cluster's pixels
        pixel_in_cluster = [None for k_i in range(k)]

        for pixel_itr, pixel in enumerate(image_vectors):

            #repeating each pixel with number of clusters mean rows for subtraction later
            pixel_nxd = np.repeat(pixel, k).reshape(3, k).T
          
            nearest_mean = np.argmin(np.linalg.norm(pixel_nxd - mean, axis=1))
            
            pixel_to_mean[pixel_itr] = nearest_mean

            if (pixel_in_cluster[nearest_mean] is None):
                pixel_in_cluster[nearest_mean] = []

            pixel_in_cluster[nearest_mean].append(pixel)

       
        for k_i in range(k):
            if (pixel_in_cluster[k_i] is not None):
                new_cluster_prototype = np.asarray(pixel_in_cluster[k_i]).sum(axis=0) / len(pixel_in_cluster[k_i])
                mean[k_i] = new_cluster_prototype

    return (pixel_to_mean, mean)

if __name__ == '__main__':
    
    f_name1 = 'Koala.jpg'
    f_name2 = 'Penguins.jpg'
    
    cluster_count= [2,5,10,15,20]
    
    #reading each pixel for koala.jpg
    image_pixels_koala = io.imread(f_name1)[:, :, :3]
    #reading each pixel for Penguins.jpg
    image_pixels_penguins = io.imread(f_name2)[:, :, :3] 
    
    for c in cluster_count:
        
        #converting each pixel to equivalent float value
        image_float = img_as_float(image_pixels_koala)
    
        image_dimensions = image_float.shape

        #reshaping nxd array to 1xd array
        image_vectors = image_float.reshape(-1, image_float.shape[-1])


        pixel_to_mean, mean = clustering(image_vectors, k=c)

        output_image = np.zeros(image_vectors.shape)
    
        for i in range(output_image.shape[0]):
            output_image[i] = mean[pixel_to_mean[i]]
    
    
        output_image = output_image.reshape(image_dimensions)
        plt.imshow(output_image)
        plt.xlabel('Koala_k_'+str(c)+'.jpg',fontsize=16)
        plt.show() 
        io.imsave('Koala_k_'+str(c)+'.jpg', output_image)
    
    for c in cluster_count:
        #converting each pixel to equivalent float value
        image_float = img_as_float(image_pixels_penguins)
    
        image_dimensions = image_float.shape

        #reshaping nxd array to 1xd array
        image_vectors = image_float.reshape(-1, image_float.shape[-1])


        pixel_to_mean, mean = clustering(image_vectors, k=c)

        output_image = np.zeros(image_vectors.shape)
    
        for i in range(output_image.shape[0]):
            output_image[i] = mean[pixel_to_mean[i]]
    
    
        output_image = output_image.reshape(image_dimensions)
        plt.imshow(output_image)
        plt.xlabel('Penguins_k_'+str(c)+'.jpg',fontsize=16)
        plt.show() 
        io.imsave('Penguins_k_'+str(c)+'.jpg', output_image)
   