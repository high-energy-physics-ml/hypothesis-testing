#Purpose: This is a DNN classification code for the EFT kinematic dataset
#Original Source: Charanjit K. Khosa, University of Genova, Italy
#Modified By: Michael Soughton, University of Sussex, UK, Michael Soughton, University of Sussex, UK
#Date: 26.03.2021

import sys, os
import numpy as np
import pandas as pd
from numpy import expand_dims

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data_dir = 'Data/'

vh_chwzero_df = pd.read_csv(data_dir + 'vh_chw_zero_100k.dat', sep="\s+", header=None)
vh_chwzp005_df = pd.read_csv(data_dir + 'vh_chw_zp005.dat', sep="\s+", header=None)

# Drop signal column if using 100k sample
vh_chwzero_df = vh_chwzero_df.iloc[:,:-1]
vh_chwzp005_df = vh_chwzp005_df.iloc[:,:-1]

# Normalising together
scaler = preprocessing.MinMaxScaler()
vh_mixed_combined_df = vh_chwzero_df.append(vh_chwzp005_df)
vh_mixed_combined_df_normalised = pd.DataFrame(scaler.fit_transform(vh_mixed_combined_df),
                             columns=vh_mixed_combined_df.columns,
                             index=vh_mixed_combined_df.index)

vh_chwzero_df_normalised = vh_mixed_combined_df_normalised.iloc[:vh_chwzero_df.shape[0],:]
vh_chwzp005_df_normalised = vh_mixed_combined_df_normalised.iloc[vh_chwzero_df.shape[0]:,:]


vh_chwzero = vh_chwzero_df_normalised.to_numpy()
vh_chwzp005 = vh_chwzp005_df_normalised.to_numpy()

# Use 100k events from each
data0 = vh_chwzero[:100000:]
data1 = vh_chwzp005[:100000:]


print("data0",data0.shape)
print('We have {} QCD jets and {} top jets'.format(len(data0), len(data1)))

# objects and labels
x_data = np.concatenate((data0, data1))
y_data = np.array([0]*len(data0)+[1]*len(data1))


print("xdatashape",x_data.shape)
y_data = keras.utils.to_categorical(y_data, 2)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)


"""
n_train = 70000
(x_train, x_test) = x_data[:n_train], x_data[n_train:]
(y_train, y_test) = y_data[:n_train], y_data[n_train:]
"""
print("x_train",x_train.shape)
print("y_train",y_train.shape)
print("x_test",x_test.shape)
#print("y_test",y_test)

model_dnn = Sequential()
model_dnn.add(Dense(20, input_dim=13, activation='relu'))
model_dnn.add(Dense(40, activation='relu'))
model_dnn.add(Dense(40, activation='relu'))
model_dnn.add(Dense(20, activation='relu'))
model_dnn.add(Dense(2, activation='softmax'))

model_dnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model_dnn.fit(x_train, y_train, validation_split=0.2, epochs=3, batch_size=100, shuffle=True, verbose=1)
history = model_dnn.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=11, batch_size=100, shuffle=True, verbose=1)





import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Accuracy002.pdf')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss002.pdf')
plt.show()


#Save the model and training history
model_dir='model_dnn/'
if not os.path.isdir(model_dir): os.system('mkdir '+model_dir)
model_dnn.save(model_dir+'dnn100k_12epochs001.h5')
np.savez(model_dir+'training_history001.npz', [history])
