
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns; sns.set(style="white", color_codes=True)


data_dir = 'Data/'
array_dir = 'bootstrap_arrays/'


extension = '_1000_bootstraps'

# To load
y_test_arr = np.load(array_dir + 'y_test_arr' + extension + '.npy')
predictions_arr = np.load(array_dir + 'predictions_arr' + extension + '.npy')
score_arr = np.load(array_dir + 'score_arr' + extension + '.npy')

# Plot score distribution because it looks nice
plt.close("all")
plt.figure()
plt.hist(score_arr, bins=40)

# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(score_arr, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(score_arr, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

# Get the top probs
top_probs_list = []
qcd_probs_list = []
for i, (predictions, y_test) in enumerate(zip(predictions_arr, y_test_arr)):
    #print("i",i)
    y_top=predictions[:,1]

    #print("y_top",y_top)
    #print("y_test",y_test)

    #ytestnew= y_test.flatten()

    ytestone = np.argmax(y_test, axis=1)
    #print("ytestone",ytestone)
    #print("y_top.shape",y_top.shape)

    top_probs = y_top[np.where(ytestone == 1)]
    qcd_probs = y_top[np.where(ytestone == 0)]

    top_probs_list.append(top_probs)
    qcd_probs_list.append(qcd_probs)

    #top_probs_arr[i,:] = top_probs

    #np.savetxt("bootstrapped_top_probs001.txt",top_probs)
    #np.savetxt("bootstrapped_qcd_probs001.txt",qcd_probs)


# Plot PDFs
nbins = 100
min_bin = min(np.min(qcd_probs), np.min(top_probs))
max_bin = min(np.max(qcd_probs), np.max(top_probs))


fig, ax = plt.subplots(1,1, figsize = (8,8))
for i, (qcd_probs, top_probs) in enumerate(zip(qcd_probs_list, top_probs_list)):
    print(i)
    ax.hist(qcd_probs, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
    ax.hist(top_probs, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.5)
    #ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
    #ax.legend()
    ax.set_xlabel('P(Top Jet)')
    ax.set_title("QCD vs Top")
    ax.set_xlim()


# Same as above but not only for good scores
qcd_pdf_list = []
top_pdf_list = []
score_list = score_arr.tolist()
fig, ax = plt.subplots(1,1, figsize = (8,8))
for i, (qcd_probs, top_probs, score) in enumerate(zip(qcd_probs_list, top_probs_list, score_list)):
    # Discard bad training
    if score > 0.88:
        print(i)
        qcd_pdf,qcd_bins,_ = ax.hist(qcd_probs, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
        top_pdf,top_bins,_ = ax.hist(top_probs, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.5)
        #ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
        #ax.legend()
        ax.set_xlabel('P(Top Jet)')
        ax.set_title("QCD vs Top")
        ax.set_xlim()
        qcd_pdf_list.append(qcd_pdf)
        top_pdf_list.append(top_pdf)

# Find average PDF
qcd_pdf_arr = np.stack((qcd_pdf_list))
top_pdf_arr = np.stack((top_pdf_list))
average_qcd_pdf_list = []
std_qcd_pdf_list = []
average_top_pdf_list = []
std_top_pdf_list = []
for i in range(qcd_pdf_arr.shape[1]):
    average_qcd_pdf = np.average(qcd_pdf_arr[:,i])
    average_qcd_pdf_list.append(average_qcd_pdf)
    std_qcd_pdf = np.sqrt(np.var(qcd_pdf_arr[:,i]))
    std_qcd_pdf_list.append(std_qcd_pdf)

    average_top_pdf = np.average(top_pdf_arr[:,i])
    average_top_pdf_list.append(average_top_pdf)
    std_top_pdf = np.sqrt(np.var(top_pdf_arr[:,i]))
    std_top_pdf_list.append(std_top_pdf)

average_qcd_pdf = np.stack((average_qcd_pdf_list))
average_top_pdf = np.stack((average_top_pdf_list))
std_qcd_pdf = np.stack((std_qcd_pdf_list))
std_top_pdf = np.stack((std_top_pdf_list))

# Center bins
qcd_bins_centered = np.zeros(len(qcd_bins) - 1)
for i in range(len(qcd_bins) - 1):
    qcd_bins_centered[i] = (qcd_bins[i] + qcd_bins[i+1])/2

top_bins_centered = np.zeros(len(top_bins) - 1)
for i in range(len(top_bins) - 1):
    top_bins_centered[i] = (top_bins[i] + top_bins[i+1])/2

# Plot averaged pdfs
fig, ax = plt.subplots(1,1, figsize = (8,8))
#ax.hist(qcd_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
#ax.hist(mixed_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.bar(qcd_bins_centered, average_qcd_pdf, yerr=std_qcd_pdf, width=np.diff(qcd_bins_centered)[0], label = 'Average QCD PDF', alpha = 0.7)
ax.bar(top_bins_centered, average_top_pdf, yerr=std_top_pdf, width=np.diff(qcd_bins_centered)[0], label = 'Average Top PDF', alpha = 0.7)
ax.legend()

# Hack the values back into prob values - I do this because the matplotlib bar chart looks a lot worse than the hist plot
# Actually never mind it looks nice now


# ============================== The same again but for smeared data ===========
"""
#array_dir = 'bootstrap_arrays/'
extension = 'zp5smeared_1000_bootstraps'

# To load
y_test_arr = np.load(array_dir + 'y_test_arr' + extension + '.npy')
predictions_arr = np.load(array_dir + 'predictions_arr' + extension + '.npy')
score_arr = np.load(array_dir + 'score_arr' + extension + '.npy')

# Plot score distribution because it looks nice
pyplot.figure()
pyplot.hist(score_arr, bins=40)

# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(score_arr, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(score_arr, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

# Get the smeared_top probs
smeared_top_probs_list = []
smeared_qcd_probs_list = []
for i, (predictions, y_test) in enumerate(zip(predictions_arr, y_test_arr)):
    #print("i",i, predictions, y_test)
    y_smeared_top=predictions[:,1]

    #print("y_smeared_top",y_smeared_top)
    #print("y_test",y_test)

    #ytestnew= y_test.flatten()

    ytestone = np.argmax(y_test, axis=1)
    #print("ytestone",ytestone)
    #print("y_smeared_top.shape",y_smeared_top.shape)

    smeared_top_probs = y_smeared_top[np.where(ytestone == 1)]
    smeared_qcd_probs = y_smeared_top[np.where(ytestone == 0)]

    smeared_top_probs_list.append(smeared_top_probs)
    smeared_qcd_probs_list.append(smeared_qcd_probs)

    #smeared_top_probs_arr[i,:] = smeared_top_probs

    #np.savetxt("bootstrapped_smeared_top_probs001.txt",smeared_top_probs)
    #np.savetxt("bootstrapped_smeared_qcd_probs001.txt",smeared_qcd_probs)


# Plot PDFs
nbins = 100
min_bin = min(np.min(smeared_qcd_probs), np.min(smeared_top_probs))
max_bin = min(np.max(smeared_qcd_probs), np.max(smeared_top_probs))

fig, ax = plt.subplots(1,1, figsize = (8,8))
for i, (smeared_qcd_probs, smeared_top_probs) in enumerate(zip(smeared_qcd_probs_list, smeared_top_probs_list)):
    print(i)
    ax.hist(smeared_qcd_probs, bins = np.linspace(0, max_bin, nbins), label = 'smeared_qcd', density = True, alpha = 0.5)
    ax.hist(smeared_top_probs, bins = np.linspace(0, max_bin, nbins), label = 'smeared_top', density = True, alpha = 0.5)
    #ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
    #ax.legend()
    ax.set_xlabel('P(smeared_top Jet)')
    ax.set_title("smeared_qcd vs smeared_top")
    ax.set_xlim()


# Same as above but not only for good scores
smeared_qcd_pdf_list = []
smeared_top_pdf_list = []
score_list = score_arr.tolist()
fig, ax = plt.subplots(1,1, figsize = (8,8))
for i, (smeared_qcd_probs, smeared_top_probs, score) in enumerate(zip(smeared_qcd_probs_list, smeared_top_probs_list, score_list)):
    # Discard bad training
    if score > 0.88:
        print(i)
        smeared_qcd_pdf,smeared_qcd_bins,_ = ax.hist(smeared_qcd_probs, bins = np.linspace(0, max_bin, nbins), label = 'smeared_qcd', density = True, alpha = 0.5)
        smeared_top_pdf,smeared_top_bins,_ = ax.hist(smeared_top_probs, bins = np.linspace(0, max_bin, nbins), label = 'smeared_top', density = True, alpha = 0.5)
        #ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
        #ax.legend()
        ax.set_xlabel('P(smeared_top Jet)')
        ax.set_title("smeared_qcd vs smeared_top")
        ax.set_xlim()
        smeared_qcd_pdf_list.append(smeared_qcd_pdf)
        smeared_top_pdf_list.append(smeared_top_pdf)

# Find average PDF
smeared_qcd_pdf_arr = np.stack((smeared_qcd_pdf_list))
smeared_top_pdf_arr = np.stack((smeared_top_pdf_list))
average_smeared_qcd_pdf_list = []
std_smeared_qcd_pdf_list = []
average_smeared_top_pdf_list = []
std_smeared_top_pdf_list = []
for i in range(smeared_qcd_pdf_arr.shape[1]):
    average_smeared_qcd_pdf = np.average(smeared_qcd_pdf_arr[:,i])
    average_smeared_qcd_pdf_list.append(average_smeared_qcd_pdf)
    std_smeared_qcd_pdf = np.sqrt(np.var(smeared_qcd_pdf_arr[:,i]))
    std_smeared_qcd_pdf_list.append(std_smeared_qcd_pdf)

    average_smeared_top_pdf = np.average(smeared_top_pdf_arr[:,i])
    average_smeared_top_pdf_list.append(average_smeared_top_pdf)
    std_smeared_top_pdf = np.sqrt(np.var(smeared_top_pdf_arr[:,i]))
    std_smeared_top_pdf_list.append(std_smeared_top_pdf)

average_smeared_qcd_pdf = np.stack((average_smeared_qcd_pdf_list))
average_smeared_top_pdf = np.stack((average_smeared_top_pdf_list))
std_smeared_qcd_pdf = np.stack((std_smeared_qcd_pdf_list))
std_smeared_top_pdf = np.stack((std_smeared_top_pdf_list))

# Center bins
smeared_qcd_bins_centered = np.zeros(len(smeared_qcd_bins) - 1)
for i in range(len(smeared_qcd_bins) - 1):
    smeared_qcd_bins_centered[i] = (smeared_qcd_bins[i] + smeared_qcd_bins[i+1])/2

smeared_top_bins_centered = np.zeros(len(smeared_top_bins) - 1)
for i in range(len(smeared_top_bins) - 1):
    smeared_top_bins_centered[i] = (smeared_top_bins[i] + smeared_top_bins[i+1])/2

# Plot averaged pdfs
fig, ax = plt.subplots(1,1, figsize = (8,8))
#ax.hist(smeared_qcd_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'smeared_qcd', density = True, alpha = 0.5)
#ax.hist(mixed_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.bar(smeared_qcd_bins_centered, average_smeared_qcd_pdf, yerr=std_smeared_qcd_pdf, width=np.diff(smeared_qcd_bins_centered)[0], label = 'Average smeared_qcd PDF', alpha = 0.7)
ax.bar(smeared_top_bins_centered, average_smeared_top_pdf, yerr=std_smeared_top_pdf, width=np.diff(smeared_qcd_bins_centered)[0], label = 'Average smeared_top PDF', alpha = 0.7)
ax.legend()
"""

# =============================== Save arrays ==================================
"""
np.savetxt("average_qcd_pdf_1000bootstraps_" + str(nbins) + "bins001.txt",average_qcd_pdf)
np.savetxt("average_top_pdf_1000bootstraps_" + str(nbins) + "bins001.txt",average_top_pdf)
np.savetxt("qcd_bins_centered_1000bootstraps_" + str(nbins) + "bins001.txt",qcd_bins_centered)
np.savetxt("top_bins_centered_1000bootstraps_" + str(nbins) + "bins001.txt",top_bins_centered)

np.savetxt("average_qcd_pdf_both_zp5smeared_1000bootstraps_" + str(nbins) + "bins001.txt",average_smeared_qcd_pdf)
np.savetxt("average_top_pdf_both_zp5smeared_1000bootstraps_" + str(nbins) + "bins001.txt",average_smeared_top_pdf)
np.savetxt("qcd_bins_centered_both_zp5smeared_1000bootstraps_" + str(nbins) + "bins001.txt",smeared_qcd_bins_centered)
np.savetxt("top_bins_centered_both_zp5smeared_1000bootstraps_" + str(nbins) + "bins001.txt",smeared_top_bins_centered)
"""

plt.ion()
plt.show()
