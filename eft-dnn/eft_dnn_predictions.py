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
#vh_chwzpz3_df = pd.read_csv(data_dir + 'vh_chwzpz3.dat', sep="\s+", header=None)

# Drop signal column if using 100k sample
vh_chwzero_df = vh_chwzero_df.iloc[:,:-1]

vh_chwzp005_df = vh_chwzp005_df.iloc[:,:-1]

# Quick renaming zp005 - > zpz3 for speed
vh_chwzpz3_df = vh_chwzp005_df

# Normalising together
scaler = preprocessing.MinMaxScaler()
vh_mixed_combined_df = vh_chwzero_df.append(vh_chwzpz3_df)
vh_mixed_combined_df_normalised = pd.DataFrame(scaler.fit_transform(vh_mixed_combined_df),
                             columns=vh_mixed_combined_df.columns,
                             index=vh_mixed_combined_df.index)

vh_chwzero_df_normalised = vh_mixed_combined_df_normalised.iloc[:vh_chwzero_df.shape[0],:]
vh_chwzpz3_df_normalised = vh_mixed_combined_df_normalised.iloc[vh_chwzero_df.shape[0]:,:]


vh_chwzero = vh_chwzero_df_normalised.to_numpy()
vh_chwzpz3 = vh_chwzpz3_df_normalised.to_numpy()

# Use 100k events from each
data0 = vh_chwzero[:100000:]
data1 = vh_chwzpz3[:100000:]


print("data0",data0.shape)
print('We have {} QCD jets and {} top jets'.format(len(data0), len(data1)))

# objects and labels
x_data = np.concatenate((data0, data1))
y_data = np.array([0]*len(data0)+[1]*len(data1))


print("xdatashape",x_data.shape)
#y_data = keras.utils.to_categorical(y_data, 2)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)



print("x_train",x_train.shape)
ytestone=y_test

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


import warnings
import matplotlib.pyplot as plt

model_dir='model_dnn/'

#history_cnn = np.load(model_dir+'training_histories.npz')['arr_0']
model_cnn = keras.models.load_model(model_dir+'dnn_100k_11epochs001.h5')

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


np.savetxt("vh_chw_zero001.txt",qcd_probs)
np.savetxt("vh_chw_zp005001.txt",top_probs)

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
