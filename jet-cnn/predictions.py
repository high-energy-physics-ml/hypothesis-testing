#Purpose: This is a CNN classification code for the jet-images data set
#Original Source: Charanjit K. Khosa, University of Genova, Italy
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

#I want to use 60K events from each sample (total-x=30K)
data0 = np.delete(data0,np.s_[1:714],0)
data1 = np.delete(data1,np.s_[1:1762],0)

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


# shapeuffle
np.random.seed(4) # for reproducibility
x_data, y_data = np.random.permutation(np.array([x_data, y_data]).T).T

# the data coming out of previous commands is a list of 2D arrays. We want a 3D np array (n_events, xpixels, ypixels)
x_data = np.stack(x_data)
#y_data= np.stack(y_data)

print("xshape-after stack",x_data.shape)
#print("x-after stack",x_data)

x_data=x_data /255.
x_data = expand_dims(x_data, axis=3)
#print("xdatashape-afterNorm255",x_data[1][0][10:21][:])


#y_data = keras.utils.to_categorical(y_data, 2)


n_train = 80000
(x_train, x_test) = x_data[:n_train], x_data[n_train:]
(y_train, y_test) = y_data[:n_train], y_data[n_train:]

print("x_train",x_train.shape)
ytestone=y_test

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


import warnings
import matplotlib.pyplot as plt

model_dir='model_cnn/'

#history_cnn = np.load(model_dir+'training_histories.npz')['arr_0']
model_cnn = keras.models.load_model(model_dir+'cnn005.h5')

predictions_cnn = model_cnn.predict(x_test)

#print("predictions_cnn",predictions_cnn[10])



from sklearn.metrics import roc_curve

fpr_cnn, tpr_cnn, thresholds = roc_curve(y_test.ravel(), predictions_cnn.ravel())

from sklearn.metrics import auc

auc = auc(fpr_cnn, tpr_cnn)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_cnn, tpr_cnn, label='(AUC = {:.3f})'.format(auc))
plt.gca().set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve', xlim=(-0.01,1.01), ylim=(-0.01,1.01))
plt.grid(True, which="both")
plt.legend(loc='lower right');
plt.savefig('ROC_curve.png')


y_top=predictions_cnn[:,1]

print("y_top",y_top)
print("y_test",y_test)

#ytestnew= y_test.flatten()

print("ytestone",ytestone)
print("y_top.shape",y_top.shape)

top_probs = y_top[np.where(ytestone == 1)]
qcd_probs = y_top[np.where(ytestone == 0)]


np.savetxt("top_probs005.txt",top_probs)
np.savetxt("qcd_probs005.txt",qcd_probs)

print("top_probs",top_probs)

import seaborn as sns; sns.set(style="white", color_codes=True)
# Make KDE plot
fig, ax = plt.subplots(figsize=(8, 8))
#ax = plt.gca()
susy_pdf_plot = sns.kdeplot(top_probs,label="Top")
other_sig_pdf_plot = sns.kdeplot(qcd_probs,label="QCD")
ax.set_title("Top vs QCD")
ax.set_xlabel(r"P(Top)")
ax.set_ylabel("PDF")
ax.set_yticks([])
ax.legend()
plt.savefig("PDF_topqcd.pdf")
