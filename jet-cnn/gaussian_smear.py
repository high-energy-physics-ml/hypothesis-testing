import sys, os
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
from skimage import filters

plt.close("all")

def pad_image(image, max_size = (25,25)):
    """
    Simply pad an image with zeros up to max_size.
    """
    size = np.shape(image)
    px, py = (max_size[0]-size[0]), (max_size[1]-size[1])
    a1=int(np.floor(px/2.0))
    a2=int(np.ceil(px/2.0))
    a3=int(np.floor(py/2.0))
    a4=int(np.ceil(py/2.0))
    image = np.pad(image, ((a1, a2), (a3, a4)), 'constant', constant_values=(0))
    #image = np.pad(image, (map(int,((np.floor(px/2.), np.ceil(px/2.)))), map(int,(np.floor(py/2.), np.ceil(py/2.)))), 'constant')
    return image

def normalize(histo, multi=255):
    """
    Normalize picture in [0,multi] range, with integer steps. E.g. multi=255 for 256 steps.
    """
    return (histo/np.max(histo)*multi).astype(int)

data_dir = 'Data/'

#Loading input data
data0 = np.load(data_dir + 'qcd_leading_jet.npz',allow_pickle=True,encoding = 'latin1')['arr_0']
data1 = np.load(data_dir + 'top_leading_jet.npz',allow_pickle=True,encoding = 'latin1')['arr_0']

print("data0",data0.shape)
print("data1",data1.shape)

#I want to use 50K events from each sample (total-x=40K)
data0 = np.delete(data0,np.s_[1:10714],0)
data1 = np.delete(data1,np.s_[1:11762],0)

print("data0",data0.shape)
print('We have {} QCD jets and {} top jets'.format(len(data0), len(data1)))

# objects and labels
x_data = np.concatenate((data0, data1))
y_data = np.array([0]*len(data0)+[1]*len(data1))


print("xdatashape",x_data.shape)

# pad and normalize images
x_data = list(map(pad_image, x_data))
#print("xdatashape",x_data.shape)
x_data = list(map(normalize, x_data))
print("xdatashape-afterNorm",x_data[1][17:21][:])

print("xdatashape-before-reshuffle",len(x_data))
# shapeuffle
np.random.seed(0) # for reproducibility
x_data, y_data = np.random.permutation(np.array([x_data, y_data]).T).T



# the data coming out of previous commands is a list of 2D arrays. We want a 3D np array (n_events, xpixels, ypixels)
x_data = np.stack(x_data)
#y_data= np.stack(y_data)

print("xshape-after stack",x_data.shape)
#print("x-after stack",x_data)

x_data=x_data /255.
x_data = expand_dims(x_data, axis=3)

# Show an image
img = x_data[0,:,:,0]
plt.figure()
plt.imshow(img)

# Apply Gaussian blur, creating a new image
sigma = 5.0
blurred_img = filters.gaussian(img, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

plt.figure()
plt.imshow(blurred_img)

# Function to blur all images within a dataset
def blur_images(images, sigma):
    blurred_images = []
    for image in images:
        blurred_image = filters.gaussian(image, sigma=(sigma,sigma), truncate=3.5, multichannel=True)
        # Renormalise blurred images
        blurred_image = blurred_image/np.max(blurred_image)
        blurred_images.append(blurred_image)
    return np.array(blurred_images)

def blur_images_at_indices(images, indices, sigma):
    images_copy = np.stack(list(images))
    for idx,image in enumerate(images_copy):
        if idx in indices:
            blurred_image = filters.gaussian(image, sigma=(sigma,sigma), truncate=3.5, multichannel=True)
            print("blurring image",idx)
            images_copy[idx,:,:,:] = blurred_image
            # Renormalise blurred images
            images_copy[idx,:,:,:] = images_copy[idx,:,:,:]/np.max(images_copy[idx,:,:,:])
    return images_copy

blurred_x_data = blur_images(x_data, 1.0)

for i in range(3):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(x_data[i,:,:,0])
    ax2.imshow(blurred_x_data[i,:,:,0])


blurred_at_idx = blur_images_at_indices(x_data, [0,3], 1.0)

"""
plt.figure()
plt.imshow(blurred_x_data[0,:,:,0])

plt.figure()
plt.imshow(blurred_at_idx[0,:,:,0])
plt.figure()
plt.imshow(blurred_at_idx[1,:,:,0])
plt.figure()
plt.imshow(blurred_at_idx[2,:,:,0])
plt.figure()
plt.imshow(blurred_at_idx[3,:,:,0])
"""
