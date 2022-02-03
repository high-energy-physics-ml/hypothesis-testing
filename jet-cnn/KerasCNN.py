
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

data_dir = 'Data/'

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


n_train = 80000
(x_train, x_test) = x_data[:n_train], x_data[n_train:]
(y_train, y_test) = y_data[:n_train], y_data[n_train:]

print("x_train",x_train.shape)
print("y_train",y_train.shape)
print("x_test",x_test.shape)
#print("y_test",y_test)


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
#history = model_cnn.fit(x_train, y_train, validation_split=0.2, epochs=3, batch_size=100, shuffle=True, verbose=1)
history = model_cnn.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=20, batch_size=100, shuffle=True, verbose=1)

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
model_cnn.save(model_dir+'cnn005.h5')
np.savez(model_dir+'training_history005.npz', [history])
