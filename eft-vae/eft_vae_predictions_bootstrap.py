import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics

import scipy.optimize
from scipy.stats import norm
from scipy import integrate

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import os
import random

plt.close("all")

# =========================== Load and prepare data ============================

data_dir = 'Data/'
plot_dir = 'Plots_bootstrap/'
model_dir = 'models_bootstrap/'

model_option = "save"

fig_specification = 'regular_EFT_run2_combnorm'
if not os.path.isdir(plot_dir): os.system('mkdir '+ plot_dir)
if not os.path.isdir(model_dir): os.system('mkdir '+ model_dir)

plt.close("all")


vh_chw_zero_df = pd.read_csv(data_dir + 'vh_chw_zero_100k.dat', sep="\s+", header=None)
vh_chw_zp005_df = pd.read_csv(data_dir + 'vh_chw_zp005.dat', sep="\s+", header=None)
vh_chw_zpz1_df = pd.read_csv(data_dir + 'vh_chw_zpz1.dat', sep="\s+", header=None)
vh_chw_zpz3_df = pd.read_csv(data_dir + 'vh_chw_zpz3.dat', sep="\s+", header=None)
vh_chw_zp1_df = pd.read_csv(data_dir + 'vh_chw_zp1.dat', sep="\s+", header=None)

# Drop signal column
vh_chw_zero_df = vh_chw_zero_df.iloc[:,:-1]
vh_chw_zp005_df = vh_chw_zp005_df.iloc[:,:-1]
vh_chw_zpz1_df = vh_chw_zpz1_df.iloc[:,:-1]
#vh_chw_zpz3_df = vh_chw_zpz3_df.iloc[:,:-1]
vh_chw_zp1_df = vh_chw_zp1_df.iloc[:,:-1]

scaler = preprocessing.MinMaxScaler()

# Normalising SM and cHW 0.005 together
vh_mixed_combined_df = vh_chw_zero_df.append(vh_chw_zp005_df)
vh_mixed_combined_df_normalised = pd.DataFrame(scaler.fit_transform(vh_mixed_combined_df),
                             columns=vh_mixed_combined_df.columns,
                             index=vh_mixed_combined_df.index)

vh_chw_zero_df_normalised = vh_mixed_combined_df_normalised.iloc[:vh_chw_zero_df.shape[0],:]
vh_chw_zp005_df_normalised = vh_mixed_combined_df_normalised.iloc[vh_chw_zero_df.shape[0]:,:]

vh_chw_zero = vh_chw_zero_df_normalised.to_numpy()
vh_chw_zp005 = vh_chw_zp005_df_normalised.to_numpy()

#x_train, x_test = train_test_split(vh_chw_zero, test_size=0.3, random_state=42)
#x_test_vh_chw_zp005 = vh_chw_zp005


"""
# Normalising all together
vh_mixed_combined_df = vh_chw_zero_df.append(vh_chw_zpz1_df)
vh_mixed_combined_df = vh_mixed_combined_df.append(vh_chw_zpz3_df)
vh_mixed_combined_df = vh_mixed_combined_df.append(vh_chw_zp1_df)
vh_mixed_combined_df_normalised = pd.DataFrame(scaler.fit_transform(vh_mixed_combined_df),
                             columns=vh_mixed_combined_df.columns,
                             index=vh_mixed_combined_df.index)

vh_chw_zero_df_normalised = vh_mixed_combined_df_normalised.iloc[:vh_chw_zero_df.shape[0],:]
vh_chw_zpz1_df_normalised = vh_mixed_combined_df_normalised.iloc[vh_chw_zero_df.shape[0]:(vh_chw_zero_df.shape[0]+vh_chw_zpz1_df.shape[0]),:]
vh_chw_zpz3_df_normalised = vh_mixed_combined_df_normalised.iloc[(vh_chw_zero_df.shape[0]+vh_chw_zpz1_df.shape[0]):(vh_chw_zero_df.shape[0]+vh_chw_zpz1_df.shape[0]+vh_chw_zpz3_df.shape[0]),:]
vh_chw_zp1_df_normalised = vh_mixed_combined_df_normalised.iloc[(vh_chw_zero_df.shape[0]+vh_chw_zpz1_df.shape[0]+vh_chw_zpz3_df.shape[0]):,:]

vh_chw_zero = vh_chw_zero_df_normalised.to_numpy()
vh_chw_zpz1 = vh_chw_zpz1_df_normalised.to_numpy()
vh_chw_zpz3 = vh_chw_zpz3_df_normalised.to_numpy()
vh_chw_zp1 = vh_chw_zp1_df_normalised.to_numpy()


x_train, x_test = train_test_split(vh_chw_zero, test_size=0.3, random_state=42)
x_test_vh_chw_zpz1 = vh_chw_zpz1
x_test_vh_chw_zpz3 = vh_chw_zpz3
x_test_vh_chw_zp1 = vh_chw_zp1
"""
"""
#to see what the results look like if training on EFT instead
x_train, x_test = train_test_split(vh_chw_zpz3, test_size=0.3, random_state=42)
x_test_vh_chw_zpz3 = vh_chw_zero
"""


# Normalise


# =========================== Model settings ===================================

# Model settings
batch_size = 256
original_shape = vh_chw_zero.shape[1:]
original_dim = np.prod(original_shape)
latent_dim = 2
intermediate_dim = 10
final_dim = original_dim
epochs = 50
epsilon_std = 0.01

# ========================== Build VAE network =================================
def create_model():
    # Build model
    #in_layer = Input(shape=original_shape)
    #x = Flatten()(in_layer)


    x = Input(shape=(original_dim,))
    #h = Dense(intermediate_dim, activation='relu')(x)
    #h = Dense(final_dim, activation = 'relu')(h)

    """
    h = Dense(intermediate_dim, activation='relu')(x)
    h = Dense(final_dim, activation = 'relu')(h)
    """
    h = Dense(final_dim, activation = 'relu')(x)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_f = Dense(final_dim, activation='relu')
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')

    #f_decoded = decoder_f(z)
    #h_decoded = decoder_h(f_decoded)

    """
    f_decoded = decoder_f(z)
    h_decoded = decoder_h(f_decoded)
    x_decoded_mean = decoder_mean(h_decoded)
    x_decoded_img = Reshape(original_shape)(x_decoded_mean)
    """
    f_decoded = decoder_f(z)
    x_decoded_mean = decoder_mean(f_decoded)
    x_decoded_img = Reshape(original_shape)(x_decoded_mean)


    # instantiate VAE model
    #vae = Model(in_layer, x_decoded_img)
    vae = Model(x, x_decoded_img)


    # Compute VAE loss
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) # is using original_dim an arbitrary choice?
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    return vae

# ============================== Train VAE =====================================
model_mse = lambda x: np.mean(np.square(x-vae.predict(x)), axis=1)
x_train_reconerror_list = []
x_test_reconerror_list = []
x_test_vh_chw_zp005_reconerror_list = []

if model_option == "save":
    n_iterations = 1000
    for i in range(n_iterations):

        x_train, x_test = train_test_split(vh_chw_zero, test_size=0.3)
        x_test_vh_chw_zp005 = vh_chw_zp005

        vae = create_model()
        print("bootstrap iteration", i+1, "/", n_iterations)

        history = vae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
                # The validation_data previously used x_test_vh_chw_zpz3 - why would the anomaly data be used for the validation data?
                # We still seems to be able to train well using x_test as validation data but it works just slightly better with
                # x_test_vh_chw_zpz3 - however how could we use it in practice if we are wanting to find anomalies?

        # Plot training losses
        #plt.figure()
        #plt.plot(history.history['loss'],label="Loss")
        #plt.plot(history.history['val_loss'],label="Validation loss")
        #plt.title("Training loss qcd-top")
        #plt.xlabel("Epochs")
        #plt.ylabel("Loss")
        #plt.yscale("log")
        #plt.legend(loc="best")

        #vae.save_weights(model_dir + 'chw_zero_trained_model2.h5')

        x_train_reconerror = model_mse(x_train)
        x_test_reconerror = model_mse(x_test)
        x_test_vh_chw_zp005_reconerror = model_mse(x_test_vh_chw_zp005)

        x_train_reconerror_list.append(x_train_reconerror)
        x_test_reconerror_list.append(x_test_reconerror)
        x_test_vh_chw_zp005_reconerror_list.append(x_test_vh_chw_zp005_reconerror)

        # Clear model and memory
        from keras import backend as K
        import gc
        del vae
        K.clear_session()
        gc.collect()
        print("Cleared session and memory")

if model_option == "load":
    from keras.models import load_model
    vae.load_weights(model_dir + 'chw_zero_trained_model2.h5')

# ============================ Plot latent space================================
"""
# Build a model to project model inputs on the latent space and plot them
encoder = Model(in_layer, z_mean) # We can defince encoder as a Model and then predict using it since we have already trained the Model 'vae'
# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
anomaly_encoded = encoder.predict(x_test_vh_chw_zpz3, batch_size=batch_size)

plt.figure(figsize=(6, 6))
plt.scatter(np.concatenate([x_test_encoded[:, 0], anomaly_encoded[:,0]],0),
                           np.concatenate([x_test_encoded[:, 1], anomaly_encoded[:,1]],0),
            c=(['g']*x_test_encoded.shape[0])+['r']*anomaly_encoded.shape[0], alpha = 0.5)
plt.title("latent space")
plt.savefig("Plots/latentSpace.png")
"""
# ============================ Find reconstruction PDF =======================
print("SM LIST")
print(x_test_reconerror_list)
print("EFT LIST")
print(x_test_vh_chw_zp005_reconerror_list)

x_test_reconerror_array = np.array([])
for array in x_test_reconerror_list:
    x_test_reconerror_array = np.append(x_test_reconerror_array, array)

x_test_vh_chw_zp005_reconerror_array = np.array([])
for array in x_test_vh_chw_zp005_reconerror_list:
    x_test_vh_chw_zp005_reconerror_array = np.append(x_test_vh_chw_zp005_reconerror_array, array)


print(x_test_reconerror_array)
print(x_test_vh_chw_zp005_reconerror_array)

print(len(x_test_reconerror_list[0]))
print(len(x_test_vh_chw_zp005_reconerror_list[0]))
print(len(x_test_reconerror_array))
print(len(x_test_vh_chw_zp005_reconerror_array))

np.savetxt("vae_outputs/vh_chw_zero_recons_zp005_cHW_normalised_13output_dim_1kbootstrap001.txt",x_test_reconerror_array)
np.savetxt("vae_outputs/vh_chw_zp005_recons_zp005_cHW_normalised_13output_dim_1kbootstrap001.txt",x_test_vh_chw_zp005_reconerror_array)



plt.show()
