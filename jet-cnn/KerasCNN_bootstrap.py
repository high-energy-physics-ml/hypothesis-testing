#!/usr/bin/env python3

#Purpose: This is a CNN classification code for the jet-images data set
#Original Source: Taken from https://gist.github.com/ilmonteux
#Modified By: Charanjit K. Khosa, University of Genova, Italy, Michael Soughton, University of Sussex, UK
#Date: 09.02.2021
import sys, os
import numpy as np
from numpy import expand_dims

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

data_dir = 'Data/'
# =========================== Take in arguments ================================
import argparse

parser = argparse.ArgumentParser(description='These are the arguments that will be passed to the script')

parser.add_argument("--smear_target",
                    type=str,
                    default="neither",
                    help="str: The jet image type that is to be smeared. Either 'neither', 'top', 'qcd', 'both'. Default is 'neither'")

parser.add_argument("--sigma",
                    type=float,
                    default=0,
                    help="float: The sigma value for smearing. Default is 0.")

parser.add_argument("--n_iter",
                    type=int,
                    default=5,
                    help="int: The number of bootstrap iterations. Default is 5.")

parser.add_argument("--n_epoch",
                    type=int,
                    default=18,
                    help="int: The number of training iterations. Default is 18.")

args = parser.parse_args()

smearing = args.sigma
n_iterations = args.n_iter
n_epochs = args.n_epoch
print("Smear target = " + str(args.smear_target) + " sigma = " + str(smearing) + " n iterations = " + str(n_iterations))

# ==============================================================================

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

from skimage import filters
def blur_images_at_indices(images, indices, sigma):
    images_copy = np.stack(list(images))
    for idx,image in enumerate(images_copy):
        if idx in indices:
            blurred_image = filters.gaussian(image, sigma=(sigma,sigma), truncate=3.5, multichannel=True)
            #print("blurring image",idx)
            images_copy[idx,:,:,:] = blurred_image
            # Renormalise blurred images
            images_copy[idx,:,:,:] = images_copy[idx,:,:,:]/np.max(images_copy[idx,:,:,:])
    return images_copy

# Preparing the data is very memory intensive and won't run on the sussex cluster in a batch so prepare it beforehand
prep_data = "load"
if prep_data == "prepare":
    print("Preparing data")
    #Loading input data
    data0 = np.load(data_dir + 'qcd_leading_jet.npz',allow_pickle=True,encoding = 'latin1')['arr_0']
    data1 = np.load(data_dir + 'top_leading_jet.npz',allow_pickle=True,encoding = 'latin1')['arr_0']

    print("data0",data0.shape)
    print("data1",data1.shape)

    #I want to use 50K events from each sample (total-x=40K)
    #data0 = np.delete(data0,np.s_[1:10714],0)
    #data1 = np.delete(data1,np.s_[1:11762],0)
    data0 = data0[:60000]
    data1 = data1[:60000]

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
    #print("xdatashape-afterNorm255",x_data[1][0][10:21][:])


    y_data = keras.utils.to_categorical(y_data, 2)

    np.save("prepped_x_data", x_data)
    np.save("prepped_y_data", y_data)

if prep_data == "load":
    print("Loading data")
    x_data = np.load("prepped_x_data.npy")
    y_data = np.load("prepped_y_data.npy")

print(x_data.shape)
print(y_data.shape)

n_train = 80000
#test_size = 1 - n_train/x_data.shape[0]
test_size = 1/3
#(x_train, x_test) = x_data[:n_train], x_data[n_train:]
#(y_train, y_test) = y_data[:n_train], y_data[n_train:]




#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

#print("x_train",x_train.shape)
#print("y_train",y_train.shape)
#print("x_test",x_test.shape)
#print("y_test",y_test)


def create_model():
    model_cnn = Sequential()
    #This is a first ConV layer, with 3 by 3 filter
    model_cnn.add(Conv2D(30, (3, 3), input_shape=(25, 25, 1), activation='relu'))
    #Second layer
    model_cnn.add(Conv2D(30, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))
    #third layer
    model_cnn.add(Conv2D(40, (3, 3), padding='same', activation='relu'))
    model_cnn.add(Conv2D(40, (3, 3), padding='same', activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.3))

    model_cnn.add(Flatten())
    #flatten layer with 300 nuerons
    model_cnn.add(Dense(300, activation='relu'))
    #model_cnn.add(Dropout(0.3))
    model_cnn.add(Dense(2, activation='softmax'))

    # Compile model
    model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_cnn


# Do the bootstrap
from sklearn.metrics import accuracy_score

y_test_list = []
predictions_list = []
score_list = []
for i in range(n_iterations):
    model_cnn = create_model()
    print("bootstrap iteration", i+1)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)

    qcd_train_indices = np.where(y_train[:,0]==1)[0]
    top_train_indices = np.where(y_train[:,1]==1)[0]
    both_train_indices = np.where(y_train[:,0]>=0)[0]

    qcd_test_indices = np.where(y_test[:,0]==1)[0]
    top_test_indices = np.where(y_test[:,1]==1)[0]
    both_test_indices = np.where(y_test[:,0]>=0)[0]

    # If jet type to smear is qcd, top or both, smear them appropriately
    if args.smear_target != "neither":
        if args.smear_target == "qcd":
            train_indices = qcd_train_indices
            test_indices = qcd_test_indices
            print("Smearing QCD")
        elif args.smear_target == "top":
            train_indices = top_train_indices
            test_indices = top_test_indices
            print("Smearing Top")
        elif args.smear_target == "both":
            train_indices = both_train_indices
            test_indices = both_test_indices
            print("Smearing both")
        x_train = blur_images_at_indices(x_train, train_indices, smearing)
        x_test = blur_images_at_indices(x_test, test_indices, smearing)
    elif args.smear_target == "neither":
        print("Not smearing either")



    history = model_cnn.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=n_epochs, batch_size=100, shuffle=True, verbose=1)
    predictions_cnn = model_cnn.predict(x_test)
    y_test_list.append(y_test)
    predictions_list.append(predictions_cnn)

    score = accuracy_score(y_test, np.round(predictions_cnn))
    score_list.append(score)

    # Clear model and memory
    from keras import backend as K
    import gc
    del model_cnn
    K.clear_session()
    gc.collect()
    print("Cleared session and memory")


y_test_arr = np.stack((y_test_list))
predictions_arr = np.stack((predictions_list))
score_arr = np.stack((score_list))

array_dir = 'bootstrap_arrays/'
extension = str(args.smear_target) + '_' + str(smearing) + 'smeared_' + str(n_iterations) + '_bootstraps'
os.makedirs(array_dir, exist_ok=True)

np.save(array_dir + 'y_test_arr' + extension, y_test_arr)
np.save(array_dir + 'predictions_arr' + extension, predictions_arr)
np.save(array_dir + 'score_arr' + extension, score_arr)

# To load
y_test_arr = np.load(array_dir + 'y_test_arr' + extension + '.npy')
predictions_arr = np.load(array_dir + 'predictions_arr' + extension + '.npy')
score_arr = np.load(array_dir + 'score_arr' + extension + '.npy')

"""
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Accuracy005.pdf')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss005.pdf')
plt.show()


#Save the model and training history
model_dir='model_cnn/'
if not os.path.isdir(model_dir): os.system('mkdir '+model_dir)
model_cnn.save(model_dir+'cnn_bootstrap001.h5')
np.savez(model_dir+'training_history005.npz', [history])
"""
