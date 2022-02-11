#!/usr/bin/env python3

"""
    Script to take a pdf of prob(EFT) and compute the log-likelihood ratio to
    perform a hypothesis test
    __author__ = "Michael Soughton", "Charanjit Kaur Khosa", "Veronica Sanz"
    __email__ =
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import log
import scipy.optimize
from scipy.stats import norm
from scipy import integrate
import random
import os
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics

# =========================== Load pdf data ====================================

sm_sample = np.loadtxt("vh_chw_zero.txt", unpack=True)
eft_sample = np.loadtxt("vh_chw_zp005.txt", unpack=True)

# Rescale probabilities so that they range form 0 - 1 instead of ~0.46 - 1
sample_minimum = min(np.min(sm_sample),np.min(eft_sample))
sm_sample = (sm_sample - sample_minimum)/(1. - sample_minimum)
eft_sample = (eft_sample - sample_minimum)/(1. - sample_minimum)

detector_efficiency = 1
# =========================== Find and plot pdf ================================
plot_dir = 'testPlots/naive_selection_cut/'
fig_specification = 'newtest001'
if not os.path.isdir(plot_dir): os.system('mkdir '+ plot_dir)
plt.close("all")

nbins = 100
min_bin = min(np.min(sm_sample), np.min(eft_sample))
max_bin = min(np.max(sm_sample), np.max(eft_sample))

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(sm_sample, bins = np.linspace(0, max_bin, nbins), label = 'sm', density = True, alpha = 0.5)
ax.hist(eft_sample, bins = np.linspace(0, max_bin, nbins), label = 'eft', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(eft Jet)')
ax.set_title("Binary classification for sm vs eft")
ax.set_xlim()
#plt.savefig(plot_dir + "Firstplot" + fig_specification + ".png")

# ============================ Setup pdfs to be used ===========================

# Function to produce pdf
def get_pdf(recons_error, min_bin=0, max_bin=1, nbins=20):
    histo,bins = np.histogram(recons_error, bins=np.linspace(min_bin, max_bin, nbins), density = True)
    return histo, bins


def get_histo(recons_error, min_bin=0, max_bin=1, nbins=20):
    histo, bins= np.histogram(recons_error, bins=np.linspace(min_bin, max_bin, nbins), density = False)
    return histo, bins

# Get reference pdf and bins
sm_reference_pdf, sm_bins = get_pdf(sm_sample, 0, 1, nbins)
eft_reference_pdf, eft_bins = get_pdf(eft_sample, 0, 1, nbins)

sm_reference_hist,_ = get_histo(sm_sample, 0, 1, nbins)
eft_reference_hist,_ = get_histo(eft_sample, 0, 1, nbins)

# Center bins
sm_bins_centered = np.zeros(len(sm_bins) - 1)
for i in range(len(sm_bins) - 1):
    sm_bins_centered[i] = (sm_bins[i] + sm_bins[i+1])/2

eft_bins_centered = np.zeros(len(eft_bins) - 1)
for i in range(len(eft_bins) - 1):
    eft_bins_centered[i] = (eft_bins[i] + eft_bins[i+1])/2

# Check that the pdf we will use in the computation is the same as the one used for visualisation
fig, ax = plt.subplots(1,1, figsize = (8,8))
#plt.hist(sm_reference_pdf,sm_bins)
plt.plot(sm_bins_centered , sm_reference_pdf)
plt.plot(eft_bins_centered , eft_reference_pdf)
ax.legend()
ax.set_xlabel('P(eft Jet)')
ax.set_title("Binary classification for sm vs eft")
ax.set_xlim()
#plt.savefig(plot_dir + "secondplot" + fig_specification + ".png")

# Function to drop zeros within the pdf since log-likelihood will return NaN in such cases
def drop_zeros(sm_pdf, eft_pdf):
    # Drop values from histograms where test histogram equal to 0
    idx_to_keep = np.where(sm_pdf != 0)[0]
    sm_pdf = sm_pdf[idx_to_keep]
    eft_pdf = eft_pdf[idx_to_keep]

    # # Drop values from histograms where anomaly histogram equal to 0
    idx_to_keep = np.where(eft_pdf != 0)[0]
    sm_pdf = sm_pdf[idx_to_keep]
    eft_pdf = eft_pdf[idx_to_keep]

    return sm_pdf, eft_pdf

# In the sm and eft probs pdf there are actually no zeros (at least for the number of bins we use)
#sm_reference_pdf, eft_reference_pdf = drop_zeros(sm_reference_pdf, eft_reference_pdf)

# Center bins
sm_bins_centered = np.zeros(len(sm_bins) - 1)
for i in range(len(sm_bins) - 1):
    sm_bins_centered[i] = (sm_bins[i] + sm_bins[i+1])/2

eft_bins_centered = np.zeros(len(eft_bins) - 1)
for i in range(len(eft_bins) - 1):
    eft_bins_centered[i] = (eft_bins[i] + eft_bins[i+1])/2

#sm_cross_section = 23.941
#eft_cross_section = 28.0
sm_cross_section = 0.014009*1000
eft_cross_section = 0.017125*1000
eft_to_sm_ratio = eft_cross_section/sm_cross_section
#eft_to_sm_ratio = 0.0005

# During development, we may set a cross section ratio instead of using the actual cross sections. So get the eft cross section here
eft_cross_section = eft_to_sm_ratio*sm_cross_section

mixed_sample = eft_sample
mixed_reference_pdf = eft_reference_pdf
mixed_bins = eft_bins

# ====================== Find and plot mixed pdf ===============================

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(sm_sample, bins = np.linspace(0, max_bin, nbins), label = 'sm', density = True, alpha = 0.5)
ax.hist(eft_sample, bins = np.linspace(0, max_bin, nbins), label = 'eft', density = True, alpha = 0.5)
ax.hist(mixed_sample, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(eft Jet)')
ax.set_title("Binary classification for sm vs eft")
ax.set_xlim()
#plt.savefig(plot_dir + "pdf_plot" + fig_specification + ".png")

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(sm_sample, bins = np.linspace(0, max_bin, nbins), label = 'sm', density = True, alpha = 0.5)
ax.hist(eft_sample, bins = np.linspace(0, max_bin, nbins), label = 'eft', density = True, alpha = 0.5)
ax.hist(mixed_sample, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(eft Jet)')
ax.set_title("Binary classification for sm vs eft")
ax.set_xlim()
#plt.savefig(plot_dir + "pdfs_with_pcut" + fig_specification + ".png")

sm_bin_width = sm_bins[1] - sm_bins[0]
eft_bin_width = eft_bins[1] - eft_bins[0]

print("fraction of sm pdf before:",sm_reference_pdf.sum()*sm_bin_width)
print("fraction of eft pdf before cut:",eft_reference_pdf.sum()*eft_bin_width)


# Convert cross sections from pb to fb
#sm_cross_section = sm_cross_section*10**3
#eft_cross_section = eft_cross_section*10**3

criterion = 'criterion3'

def get_sigma(luminosity, prob_threshold, criteria_type):

    sm_reference_pdf_cut = np.stack(list(sm_reference_pdf))
    eft_reference_pdf_cut = np.stack(list(eft_reference_pdf))
    mixed_reference_pdf_cut = np.stack(list(mixed_reference_pdf))

    sm_reference_pdf_cut[np.where(sm_bins_centered < prob_threshold)[0]] = 0
    eft_reference_pdf_cut[np.where(sm_bins_centered < prob_threshold)[0]] = 0
    mixed_reference_pdf_cut[np.where(sm_bins_centered < prob_threshold)[0]] = 0

    epsilon_sm = sm_reference_pdf_cut.sum()*sm_bin_width
    epsilon_eft = eft_reference_pdf_cut.sum()*eft_bin_width
    epsilon_mixed = mixed_reference_pdf_cut.sum()*eft_bin_width

    print("EPSILONS", epsilon_sm, epsilon_eft, epsilon_mixed)

    #print("fraction of sm pdf remaining after cut:",epsilon_sm)
    #print("fraction of eft pdf remaining after cut:",epsilon_eft)

    N_eft_after_cut = epsilon_eft*eft_cross_section*luminosity
    N_sm_after_cut = epsilon_sm*sm_cross_section*luminosity

    N_pure_eft_after_cut = N_eft_after_cut - N_sm_after_cut

    sigma_criteria1 = N_pure_eft_after_cut/np.sqrt(N_sm_after_cut)

    N_SM_1sigma = N_sm_after_cut*0.1 # Probably would have to do it before cut? would i?
    N_SM_2sigma = 2.0*N_SM_1sigma

    sigma_criteria1_lower = N_pure_eft_after_cut/np.sqrt(N_sm_after_cut + N_SM_2sigma)
    sigma_criteria1_upper = N_pure_eft_after_cut/np.sqrt(N_sm_after_cut - N_SM_2sigma)

    sigma_criteria2 = N_pure_eft_after_cut/np.sqrt(N_sm_after_cut + N_pure_eft_after_cut)
    sigma_criteria2_lower = N_pure_eft_after_cut/np.sqrt(N_sm_after_cut + N_SM_2sigma  + N_pure_eft_after_cut)
    sigma_criteria2_upper = N_pure_eft_after_cut/np.sqrt(N_sm_after_cut - N_SM_2sigma  + N_pure_eft_after_cut)

    #N_sm_after_cut = epsilon_SM*SM_cross_section
    #N_pure_eft_after_cut = epsilon_pureEFT*pureEFT_cross_section

    sigma_criteria3 = np.sqrt(2.0*((N_pure_eft_after_cut + N_sm_after_cut)*np.log(1.0 + N_pure_eft_after_cut/N_sm_after_cut) - N_pure_eft_after_cut))
    sigma_criteria3_lower = np.sqrt(2.0*((N_pure_eft_after_cut + N_sm_after_cut + N_SM_2sigma)*np.log(1.0 + N_pure_eft_after_cut/(N_sm_after_cut + N_SM_2sigma)) - N_pure_eft_after_cut))
    sigma_criteria3_upper = np.sqrt(2.0*((N_pure_eft_after_cut + N_sm_after_cut - N_SM_2sigma)*np.log(1.0 + N_pure_eft_after_cut/(N_sm_after_cut - N_SM_2sigma)) - N_pure_eft_after_cut))


    #print("epsilon SM", epsilon_SM, "epsilon EFT", epsilon_EFT, "N SM after cut =", N_sm_after_cut, "N EFT after cut =", N_pure_eft_after_cut, "N pure EFT after cut =", N_pure_eft_after_cut)
    print("Sigma criteria 1=",sigma_criteria1)
    print("Sigma criteria 2=",sigma_criteria2)
    print("Sigma criteria 3=",sigma_criteria3)
    print("S", N_pure_eft_after_cut, "B", N_sm_after_cut, "T1", (N_pure_eft_after_cut + N_sm_after_cut), "T2", np.log(1.0 + N_pure_eft_after_cut/N_sm_after_cut))
    #print("Sigma criteria error =",sigma_criteria_error)

    if criteria_type == "1":
        return sigma_criteria1
    elif criteria_type == "1lower":
        return sigma_criteria1_lower
    elif criteria_type == "1upper":
        return sigma_criteria1_upper
    elif criteria_type == "2":
        return sigma_criteria2
    elif criteria_type == "2lower":
        return sigma_criteria2_lower
    elif criteria_type == "2upper":
        return sigma_criteria2_upper
    elif criteria_type == "3":
        return sigma_criteria3
    elif criteria_type == "3lower":
        return sigma_criteria3_lower
    elif criteria_type == "3upper":
        return sigma_criteria3_upper



# Z vs Neft
luminosity_arr = np.linspace(0.1,14.0,30)
prob_threshold = 0
sigma_criteria1_list = []
sigma_criteria2_list = []
sigma_criteria3_list = []
for luminosity in luminosity_arr:
    sigma_criteria1 = get_sigma(luminosity, prob_threshold, "1")
    sigma_criteria2 = get_sigma(luminosity, prob_threshold, "2")
    sigma_criteria3 = get_sigma(luminosity, prob_threshold, "3")
    sigma_criteria1_list.append(sigma_criteria1)
    sigma_criteria2_list.append(sigma_criteria2)
    sigma_criteria3_list.append(sigma_criteria3)

# Set the entry for luminosity = smallest to be zero
sigma_criteria1_list[0] = 0
sigma_criteria2_list[0] = 0
sigma_criteria3_list[0] = 0

# Convert L to Neft
N_toy_sm_events_list = []
N_toy_eft_events_list0 = []
for luminosity in luminosity_arr:
    #N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    #N_toy_eft_events = luminosity*detector_efficiency*(eft_cross_section - sm_cross_section)
    N_toy_eft_events = luminosity*detector_efficiency*(eft_cross_section)
    #N_toy_sm_events_list.append(N_toy_sm_events)
    N_toy_eft_events_list0.append(N_toy_eft_events)
fig, ax = plt.subplots(1,1, figsize = (8,8))


# Load LLR arrays
array_dir = '../arrays/'
llr_Pcut = '0.0'
extension = 'with_poisson_0Pcut_100ktoys999'
llr_luminosity_arr = np.loadtxt(array_dir + 'luminosityZvsNeft_arr' + extension + '.txt')
llr_nstdevs_arr = np.loadtxt(array_dir + 'nstdevsZvsNeft_arr' + extension + '.txt')
llr_nstdevs_no_beta_arr = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNeft_arr' + extension + '.txt')
llr_nstdevs_exact_arr = np.loadtxt(array_dir + 'nstdevs_exactZvsNeft_arr' + extension + '.txt')

#N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
#N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
N_toy_sm_events_list = []
N_toy_eft_events_list = []
for luminosity in llr_luminosity_arr:
    #N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    #N_toy_eft_events = luminosity*detector_efficiency*(eft_cross_section - sm_cross_section)
    N_toy_eft_events = luminosity*detector_efficiency*(eft_cross_section)
    #N_toy_sm_events_list.append(N_toy_sm_events)
    N_toy_eft_events_list.append(N_toy_eft_events)

#plt.axhline(y=3, color='k', linestyle='dotted',label=r'$\alpha=1.35 \times 10^{-3}$', linewidth = 3)
#plt.axhline(y=5, color='k', linestyle='dashed',label=r'$\alpha=2.87 \times 10^{-7}$', linewidth = 3)

plt.plot(N_toy_eft_events_list, llr_nstdevs_arr,label = r'$\alpha=$ Symmetric',color='m',linestyle='dashed', linewidth = 3)
plt.plot(N_toy_eft_events_list, llr_nstdevs_no_beta_arr,label = r'$\alpha=$ Asymmetric',color='m',linestyle='solid', linewidth = 3)
#plt.plot(N_toy_eft_events_list, llr_nstdevs_exact_arr,label = r'Exact')

#ax.plot(N_toy_eft_events_list0, sigma_criteria1_list, label=r'$Z = S/\sqrt{B}$',color='g',linestyle='solid')
#ax.plot(N_toy_eft_events_list0, sigma_criteria2_list, label=r'$Z = S/\sqrt{S + B}$',color='b',linestyle='dashed')
ax.plot(N_toy_eft_events_list0, sigma_criteria3_list, label=r'Z = $ Asimov',color='y',linestyle='dotted', linewidth = 3)


plt.legend()
plt.xlabel(r'$N_\mathrm{EFT}$',fontsize=18)
plt.ylabel(r'Significance', fontsize=18)

#ax.set_title(r'$P_\mathrm{cut}$(EFT) = %.1f' % prob_threshold, fontsize=18)




"""
# Z vs Pcut
luminosity = 5.0
N_eft_before_cut = eft_cross_section*luminosity
N_sm_before_cut = sm_cross_section*luminosity

prob_threshold_list = np.linspace(0,0.8,100)
sigma_criteria1_list = []
sigma_criteria2_list = []
sigma_criteria3_list = []

for prob_threshold in prob_threshold_list:
    sigma_criteria1 = get_sigma(luminosity, prob_threshold, "1")
    sigma_criteria2 = get_sigma(luminosity, prob_threshold, "2")
    sigma_criteria3 = get_sigma(luminosity, prob_threshold, "3")
    sigma_criteria1_list.append(sigma_criteria1)
    sigma_criteria2_list.append(sigma_criteria2)
    sigma_criteria3_list.append(sigma_criteria3)


# Load LLR significances
extension = '5L_002'
llr_luminosity_arr = np.loadtxt(array_dir + 'testluminosityZvsPcut_arr' + extension + '.txt')
llr_prob_threshold_arr = np.loadtxt(array_dir + 'testprob_thresholdZvsPcut_arr' + extension + '.txt')
llr_nstdevs_arr = np.loadtxt(array_dir + 'testnstdevsZvsPcut_arr' + extension + '.txt')
llr_nstdevs_exact_arr = np.loadtxt(array_dir + 'testnstdevs_exactZvsPcut_arr' + extension + '.txt')



fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(prob_threshold_list, sigma_criteria1_list, label=r'$S/\sqrt{B}$, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_eft_before_cut, N_sm_before_cut),alpha=0.5)
ax.plot(prob_threshold_list, sigma_criteria2_list, label=r'$S/\sqrt{S + B}$, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_eft_before_cut, N_sm_before_cut),alpha=0.5)
ax.plot(prob_threshold_list, sigma_criteria3_list, label=r'Asimov, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_eft_before_cut, N_sm_before_cut),alpha=0.5)

if llr_luminosity_arr.size > 1:
    for i,luminosity in enumerate(llr_luminosity_arr):
        N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
        N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
        plt.plot(llr_prob_threshold_arr, llr_nstdevs_arr[i],label = r'Approx $N_\mathrm{sm} = %s$, $N_\mathrm{eft} = %s$' % (N_toy_sm_events, N_toy_mixed_events))
        #plt.plot(llr_prob_threshold_arr, llr_nstdevs_exact_arr[i],label = r'Exact $N_\mathrm{sm} = %s$, $N_\mathrm{eft} = %s$' % (N_toy_sm_events, N_toy_sm_events))
        plt.legend()
        plt.xlabel(r'$P_{cut}$')
        plt.ylabel(r'Significance $Z$')
        #plt.ylim(0,10)
else:
    N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
    plt.plot(llr_prob_threshold_arr, llr_nstdevs_arr,label = r'Approx $N_\mathrm{sm} = %s$, $N_\mathrm{eft} = %s$' % (N_toy_sm_events, N_toy_mixed_events))
    #plt.plot(llr_prob_threshold_arr, llr_nstdevs_exact_arr,label = r'$N_\mathrm{sm} = {}, $N_\mathrm{eft} = {}$'.format(N_toy_sm_events, N_toy_sm_events))
    plt.legend()
    plt.xlabel(r'$P_{cut}$')
    plt.ylabel(r'Significance $Z$')
    #plt.ylim(0,10)

#ax.plot(llr_prob_threshold_arr, llr_nstdevs_arr, label=r'LLR')
ax.legend()
ax.set_xlabel(r'$P_\mathrm{cut}$')
ax.set_ylabel(r'$Z$')
"""









#plt.ion()
plt.show()
