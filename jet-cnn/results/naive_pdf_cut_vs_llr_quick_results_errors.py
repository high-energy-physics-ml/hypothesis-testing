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
import random
import os

# =========================== Load pdf data ====================================

# NEED TO IMPORT THE BOOTSTRAPPED FILES FOR CONSISTENCY INSTEAD









top_sample = np.loadtxt("top.txt", unpack=True)
qcd_sample = np.loadtxt("qcd.txt", unpack=True)
#top_sample = np.loadtxt("top_probs_both_smeared003.txt", unpack=True)
#qcd_sample = np.loadtxt("qcd_probs_both_smeared003", unpack=True)


detector_efficiency = 1
# =========================== Find and plot pdf ================================
plot_dir = 'testPlots/naive_selection_cut/'
fig_specification = 'newtest001'
if not os.path.isdir(plot_dir): os.system('mkdir '+ plot_dir)
plt.close("all")

nbins = 100
min_bin = min(np.min(qcd_sample), np.min(top_sample))
max_bin = min(np.max(qcd_sample), np.max(top_sample))

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(qcd_sample, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
ax.hist(top_sample, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(Top Jet)')
ax.set_title("Binary classification for QCD vs Top")
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
qcd_reference_pdf, qcd_bins = get_pdf(qcd_sample, 0, 1, nbins)
top_reference_pdf, top_bins = get_pdf(top_sample, 0, 1, nbins)

qcd_reference_hist,_ = get_histo(qcd_sample, 0, 1, nbins)
top_reference_hist,_ = get_histo(top_sample, 0, 1, nbins)

# Center bins
qcd_bins_centered = np.zeros(len(qcd_bins) - 1)
for i in range(len(qcd_bins) - 1):
    qcd_bins_centered[i] = (qcd_bins[i] + qcd_bins[i+1])/2

top_bins_centered = np.zeros(len(top_bins) - 1)
for i in range(len(top_bins) - 1):
    top_bins_centered[i] = (top_bins[i] + top_bins[i+1])/2

# Check that the pdf we will use in the computation is the same as the one used for visualisation
fig, ax = plt.subplots(1,1, figsize = (8,8))
#plt.hist(qcd_reference_pdf,qcd_bins)
plt.plot(qcd_bins_centered , qcd_reference_pdf)
plt.plot(top_bins_centered , top_reference_pdf)
ax.legend()
ax.set_xlabel('P(Top Jet)')
ax.set_title("Binary classification for QCD vs Top")
ax.set_xlim()
#plt.savefig(plot_dir + "secondplot" + fig_specification + ".png")

# Function to drop zeros within the pdf since log-likelihood will return NaN in such cases
def drop_zeros(qcd_pdf, top_pdf):
    # Drop values from histograms where test histogram equal to 0
    idx_to_keep = np.where(qcd_pdf != 0)[0]
    qcd_pdf = qcd_pdf[idx_to_keep]
    top_pdf = top_pdf[idx_to_keep]

    # # Drop values from histograms where anomaly histogram equal to 0
    idx_to_keep = np.where(top_pdf != 0)[0]
    qcd_pdf = qcd_pdf[idx_to_keep]
    top_pdf = top_pdf[idx_to_keep]

    return qcd_pdf, top_pdf

# In the qcd and top probs pdf there are actually no zeros (at least for the number of bins we use)
#qcd_reference_pdf, top_reference_pdf = drop_zeros(qcd_reference_pdf, top_reference_pdf)

# Center bins
qcd_bins_centered = np.zeros(len(qcd_bins) - 1)
for i in range(len(qcd_bins) - 1):
    qcd_bins_centered[i] = (qcd_bins[i] + qcd_bins[i+1])/2

top_bins_centered = np.zeros(len(top_bins) - 1)
for i in range(len(top_bins) - 1):
    top_bins_centered[i] = (top_bins[i] + top_bins[i+1])/2

# Make a mixed pdf whose weights we will use to sample from
def mix_pdfs(qcd_sample, histB_bins, histB, B_to_A_ratio):
    # Do we add pdf B to A, keeping A the same, then renomralising. Or do we
    # mix A and B simultaenously, keeping the normalisation = 1 throughout?
    # Let's do the first way

    # Do we do it using histos or pdfs?
    num_top_events_to_mix = int(B_to_A_ratio*len(qcd_sample))
    top_prob_values = random.choices(histB_bins, weights = histB, k = num_top_events_to_mix)
    mixed_prob_values = np.concatenate((qcd_sample, top_prob_values))

    mixed_reference_pdf, mixed_pdf_bins = get_pdf(mixed_prob_values, 0, 1, nbins)
    return mixed_prob_values, mixed_reference_pdf, mixed_pdf_bins

#top_cross_section = 0.137129
#qcd_cross_section = 58.09545
top_cross_section = 53.1684
qcd_cross_section = 48829.2
top_to_qcd_ratio = top_cross_section/qcd_cross_section
#top_to_qcd_ratio = 0.0005

# During development, we may set a cross section ratio instead of using the actual cross sections. So get the top cross section here
top_cross_section = top_to_qcd_ratio*qcd_cross_section

mixed_prob_values, mixed_reference_pdf, mixed_bins = mix_pdfs(qcd_sample, top_bins_centered, top_reference_hist, top_to_qcd_ratio)

# ====================== Find and plot mixed pdf ===============================

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(qcd_sample, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
ax.hist(top_sample, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.5)
ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(Top Jet)')
ax.set_title("Binary classification for QCD vs Top")
ax.set_xlim()
#plt.savefig(plot_dir + "pdf_plot" + fig_specification + ".png")

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(qcd_sample, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
ax.hist(top_sample, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.5)
ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(Top Jet)')
ax.set_title("Binary classification for QCD vs Top")
ax.set_xlim()
#plt.savefig(plot_dir + "pdfs_with_pcut" + fig_specification + ".png")

qcd_bin_width = qcd_bins[1] - qcd_bins[0]
top_bin_width = top_bins[1] - top_bins[0]

print("fraction of QCD pdf before:",qcd_reference_pdf.sum()*qcd_bin_width)
print("fraction of top pdf before cut:",top_reference_pdf.sum()*top_bin_width)


# Convert cross sections from pb to fb
#qcd_cross_section = qcd_cross_section*10**3
#top_cross_section = top_cross_section*10**3

criterion = 'criterion3'

def get_sigma(luminosity, prob_threshold, criteria_type, shift=0):

    qcd_reference_pdf_cut = np.stack(list(qcd_reference_pdf))
    top_reference_pdf_cut = np.stack(list(top_reference_pdf))
    mixed_reference_pdf_cut = np.stack(list(mixed_reference_pdf))

    qcd_reference_pdf_cut[np.where(qcd_bins_centered < prob_threshold)[0]] = 0
    top_reference_pdf_cut[np.where(qcd_bins_centered < prob_threshold)[0]] = 0
    mixed_reference_pdf_cut[np.where(qcd_bins_centered < prob_threshold)[0]] = 0

    epsilon_qcd = qcd_reference_pdf_cut.sum()*qcd_bin_width
    epsilon_top = top_reference_pdf_cut.sum()*top_bin_width
    epsilon_mixed = mixed_reference_pdf_cut.sum()*top_bin_width

    print("EPSILONS", epsilon_qcd, epsilon_top, epsilon_mixed)

    #print("fraction of QCD pdf remaining after cut:",epsilon_qcd)
    #print("fraction of top pdf remaining after cut:",epsilon_top)

    #N_top_after_cut = epsilon_top*top_cross_section*luminosity
    N_top_after_cut = epsilon_top*top_cross_section*luminosity
    #N_top_after_cut = detector_signal_detection_efficiency*epsilon_top*(epsilon_qcd/epsilon_mixed)*top_cross_section*luminosity
    N_qcd_after_cut = epsilon_qcd*qcd_cross_section*luminosity

    sigma_criteria1 = N_top_after_cut/np.sqrt(N_qcd_after_cut)

    N_SM_1sigma = N_qcd_after_cut*shift # Probably would have to do it before cut? would i?

    sigma_criteria1_lower = N_top_after_cut/np.sqrt(N_qcd_after_cut + N_SM_1sigma)
    sigma_criteria1_upper = N_top_after_cut/np.sqrt(N_qcd_after_cut - N_SM_1sigma)

    sigma_criteria2 = N_top_after_cut/np.sqrt(N_qcd_after_cut + N_top_after_cut)
    sigma_criteria2_lower = N_top_after_cut/np.sqrt(N_qcd_after_cut + N_SM_1sigma  + N_top_after_cut)
    sigma_criteria2_upper = N_top_after_cut/np.sqrt(N_qcd_after_cut - N_SM_1sigma  + N_top_after_cut)

    #N_qcd_after_cut = epsilon_SM*SM_cross_section
    #N_top_after_cut = epsilon_pureEFT*pureEFT_cross_section

    sigma_criteria3 = np.sqrt(2.0*((N_top_after_cut + N_qcd_after_cut)*np.log(1.0 + N_top_after_cut/N_qcd_after_cut) - N_top_after_cut))
    sigma_criteria3_lower = np.sqrt(2.0*((N_top_after_cut + N_qcd_after_cut + N_SM_1sigma)*np.log(1.0 + N_top_after_cut/(N_qcd_after_cut + N_SM_1sigma)) - N_top_after_cut))
    sigma_criteria3_upper = np.sqrt(2.0*((N_top_after_cut + N_qcd_after_cut - N_SM_1sigma)*np.log(1.0 + N_top_after_cut/(N_qcd_after_cut - N_SM_1sigma)) - N_top_after_cut))


    #print("epsilon SM", epsilon_SM, "epsilon EFT", epsilon_EFT, "N SM after cut =", N_qcd_after_cut, "N EFT after cut =", N_top_after_cut, "N pure EFT after cut =", N_top_after_cut)
    print("Sigma criteria 1=",sigma_criteria1)
    print("Sigma criteria 2=",sigma_criteria2)
    print("Sigma criteria 3=",sigma_criteria3)
    print("S", N_top_after_cut, "B", N_qcd_after_cut, "T1", (N_top_after_cut + N_qcd_after_cut), "T2", np.log(1.0 + N_top_after_cut/N_qcd_after_cut))
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



# Z vs Ntop
luminosity_arr = np.linspace(0.01,10.0,11)
prob_threshold = 0.0
sigma_criteria1_list = []
sigma_criteria2_list = []
sigma_criteria3_list = []

sigma_criteria3upper0p1_list = []
sigma_criteria3lower0p1_list = []
sigma_criteria3upper0p2_list = []
sigma_criteria3lower0p2_list = []
sigma_criteria3upper0p3_list = []
sigma_criteria3lower0p3_list = []
for luminosity in luminosity_arr:
    sigma_criteria1 = get_sigma(luminosity, prob_threshold, "1")
    sigma_criteria2 = get_sigma(luminosity, prob_threshold, "2")
    sigma_criteria3 = get_sigma(luminosity, prob_threshold, "3")
    sigma_criteria1_list.append(sigma_criteria1)
    sigma_criteria2_list.append(sigma_criteria2)
    sigma_criteria3_list.append(sigma_criteria3)

    sigma_criteria3_upper0p1 = get_sigma(luminosity, prob_threshold, "3upper",0.1)
    sigma_criteria3_lower0p1 = get_sigma(luminosity, prob_threshold, "3lower",0.1)
    sigma_criteria3_upper0p2 = get_sigma(luminosity, prob_threshold, "3upper",0.2)
    sigma_criteria3_lower0p2 = get_sigma(luminosity, prob_threshold, "3lower",0.2)
    sigma_criteria3_upper0p3 = get_sigma(luminosity, prob_threshold, "3upper",0.3)
    sigma_criteria3_lower0p3 = get_sigma(luminosity, prob_threshold, "3lower",0.3)
    sigma_criteria3upper0p1_list.append(sigma_criteria3_upper0p1)
    sigma_criteria3lower0p1_list.append(sigma_criteria3_lower0p1)
    sigma_criteria3upper0p2_list.append(sigma_criteria3_upper0p2)
    sigma_criteria3lower0p2_list.append(sigma_criteria3_lower0p2)
    sigma_criteria3upper0p3_list.append(sigma_criteria3_upper0p3)
    sigma_criteria3lower0p3_list.append(sigma_criteria3_lower0p3)

# Set first entry to zero
sigma_criteria1_list[0] = 0
sigma_criteria2_list[0] = 0
sigma_criteria3_list[0] = 0

# Convert L to Ntop
N_toy_qcd_events_list = []
N_toy_top_events_list = []
for luminosity in luminosity_arr:
    #N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
    N_toy_top_events = int(luminosity*detector_efficiency*top_cross_section)
    #N_toy_qcd_events_list.append(N_toy_qcd_events)
    N_toy_top_events_list.append(N_toy_top_events)
fig, ax = plt.subplots(1,1, figsize = (8,8))

#ax.plot(N_toy_top_events_list, sigma_criteria3_list, label=r'Asimov,  Remember that the naive significance is twice the value that ours would be compared to')


# Load LLR arrays
array_dir = '../arrays/'
llr_Pcut = '0.0'
extension0 = 'with_poisson_' + llr_Pcut + 'Pcut_100ktoys001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr0 = np.loadtxt(array_dir + 'luminosityZvsNtop_arr' + extension0 + '.txt')
llr_nstdevs_arr0 = np.loadtxt(array_dir + 'nstdevsZvsNtop_arr' + extension0 + '.txt')
llr_nstdevs_no_beta_arr0 = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension0 + '.txt')
llr_nstdevs_exact_arr0 = np.loadtxt(array_dir + 'nstdevs_exactZvsNtop_arr' + extension0 + '.txt')

extension = 'with_poisson_' + llr_Pcut + 'Pcut_100ktoys_-0.1xsec_001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr = np.loadtxt(array_dir + 'luminosityZvsNtop_arr' + extension + '.txt')
llr_nstdevs_arr = np.loadtxt(array_dir + 'nstdevsZvsNtop_arr' + extension + '.txt')
llr_nstdevs_no_beta_arr = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension + '.txt')
llr_nstdevs_exact_arr = np.loadtxt(array_dir + 'nstdevs_exactZvsNtop_arr' + extension + '.txt')

extension2 = 'with_poisson_' + llr_Pcut + 'Pcut_100ktoys_0.1xsec_001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr2 = np.loadtxt(array_dir + 'luminosityZvsNtop_arr' + extension2 + '.txt')
llr_nstdevs_arr2 = np.loadtxt(array_dir + 'nstdevsZvsNtop_arr' + extension2 + '.txt')
llr_nstdevs_no_beta_arr2 = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension2 + '.txt')
llr_nstdevs_exact_arr2 = np.loadtxt(array_dir + 'nstdevs_exactZvsNtop_arr' + extension2 + '.txt')

extension3 = 'with_poisson_' + llr_Pcut + 'Pcut_100ktoys_-0.2xsec_001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr3 = np.loadtxt(array_dir + 'luminosityZvsNtop_arr' + extension3 + '.txt')
llr_nstdevs_arr3 = np.loadtxt(array_dir + 'nstdevsZvsNtop_arr' + extension3 + '.txt')
llr_nstdevs_no_beta_arr3 = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension3 + '.txt')
llr_nstdevs_exact_arr3 = np.loadtxt(array_dir + 'nstdevs_exactZvsNtop_arr' + extension3 + '.txt')

extension4 = 'with_poisson_' + llr_Pcut + 'Pcut_100ktoys_0.2xsec_001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr4 = np.loadtxt(array_dir + 'luminosityZvsNtop_arr' + extension4 + '.txt')
llr_nstdevs_arr4 = np.loadtxt(array_dir + 'nstdevsZvsNtop_arr' + extension4 + '.txt')
llr_nstdevs_no_beta_arr4 = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension4 + '.txt')
llr_nstdevs_exact_arr4 = np.loadtxt(array_dir + 'nstdevs_exactZvsNtop_arr' + extension4 + '.txt')

extension5 = 'with_poisson_' + llr_Pcut + 'Pcut_100ktoys_-0.3xsec_001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr5 = np.loadtxt(array_dir + 'luminosityZvsNtop_arr' + extension5 + '.txt')
llr_nstdevs_arr5 = np.loadtxt(array_dir + 'nstdevsZvsNtop_arr' + extension5 + '.txt')
llr_nstdevs_no_beta_arr5 = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension5 + '.txt')
llr_nstdevs_exact_arr5 = np.loadtxt(array_dir + 'nstdevs_exactZvsNtop_arr' + extension5 + '.txt')

extension6 = 'with_poisson_' + llr_Pcut + 'Pcut_100ktoys_0.3xsec_001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr6 = np.loadtxt(array_dir + 'luminosityZvsNtop_arr' + extension6 + '.txt')
llr_nstdevs_arr6 = np.loadtxt(array_dir + 'nstdevsZvsNtop_arr' + extension6 + '.txt')
llr_nstdevs_no_beta_arr6 = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension6 + '.txt')
llr_nstdevs_exact_arr6 = np.loadtxt(array_dir + 'nstdevs_exactZvsNtop_arr' + extension6 + '.txt')


#llr_nstdevs_arr3 = 0.5*llr_nstdevs_arr3

N_toy_qcd_events_list = []
N_toy_top_events_list = []
for luminosity in llr_luminosity_arr:
    #N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
    N_toy_top_events = int(luminosity*detector_efficiency*top_cross_section)
    #N_toy_qcd_events_list.append(N_toy_qcd_events)
    N_toy_top_events_list.append(N_toy_top_events)


#plt.axhline(y=5, color='k', linestyle='--',label=r'$\alpha=2.87 \times 10^{-7}$')

#plt.plot(N_toy_top_events_list, llr_nstdevs_exact_arr,label = r'Not smeared Exact')
plt.plot(N_toy_top_events_list, llr_nstdevs_arr0,label = r'$\alpha=$ Symmetric',color='r', linestyle='dashed')
plt.plot(N_toy_top_events_list, llr_nstdevs_no_beta_arr0,label = r'$\alpha=$ Asymmetric',color='r', linestyle='solid')
#=plt.plot(N_toy_top_events_list, llr_nstdevs_arr,label = r'0 Pcut 0.1 shift')
#plt.plot(N_toy_top_events_list, llr_nstdevs_no_beta_arr,label = r'0 Pcut no beta 0.1 shift')
#plt.plot(N_toy_top_events_list, llr_nstdevs_arr2,label = r'0 Pcut -0.1 shift')
#plt.plot(N_toy_top_events_list, llr_nstdevs_no_beta_arr2,label = r'0 Pcut no beta -0.1 shift')
#plt.plot(N_toy_top_events_list, llr_nstdevs_arr3,label = r'0 Pcut 0.2 shift')
#plt.plot(N_toy_top_events_list, llr_nstdevs_no_beta_arr3,label = r'0 Pcut no beta 0.2 shift')
#plt.plot(N_toy_top_events_list, llr_nstdevs_arr4,label = r'0 Pcut -0.2 shift')
#plt.plot(N_toy_top_events_list, llr_nstdevs_no_beta_arr4,label = r'0 Pcut no beta -0.2 shift')

# Remember that the naive significance is twice the value that ours would be compared to
ax.plot(N_toy_top_events_list, sigma_criteria1_list, label=r'$Z = S/\sqrt{B}$',color='g',linestyle='solid')
ax.plot(N_toy_top_events_list, sigma_criteria2_list, label=r'$Z = S/\sqrt{S + B}$',color='b',linestyle='dashed')
ax.plot(N_toy_top_events_list, sigma_criteria3_list, label=r'$Z=$ Asimov',color='y',linestyle='dotted')

plt.fill_between(N_toy_top_events_list, sigma_criteria3lower0p1_list, sigma_criteria3upper0p1_list, alpha=0.2, color='r')
plt.fill_between(N_toy_top_events_list, sigma_criteria3lower0p2_list, sigma_criteria3upper0p2_list, alpha=0.2, color='g')
plt.fill_between(N_toy_top_events_list, sigma_criteria3lower0p3_list, sigma_criteria3upper0p3_list, alpha=0.2, color='b')



# Set the first entry in these error nstdev arrays to be 0 as should be expected
llr_nstdevs_arr[0] = 0
llr_nstdevs_no_beta_arr[0] = 0
llr_nstdevs_arr2[0] = 0
llr_nstdevs_no_beta_arr2[0] = 0
llr_nstdevs_arr3[0] = 0
llr_nstdevs_no_beta_arr3[0] = 0
llr_nstdevs_arr4[0] = 0
llr_nstdevs_no_beta_arr4[0] = 0
llr_nstdevs_arr5[0] = 0
llr_nstdevs_no_beta_arr5[0] = 0
llr_nstdevs_arr6[0] = 0
llr_nstdevs_no_beta_arr6[0] = 0


plt.fill_between(N_toy_top_events_list, llr_nstdevs_arr2, llr_nstdevs_arr, alpha=0.2, color='r', label=r'0.1 $\sigma_\mathrm{QCD}$ shift')
plt.fill_between(N_toy_top_events_list, llr_nstdevs_no_beta_arr2, llr_nstdevs_no_beta_arr, alpha=0.2, color='r')
plt.fill_between(N_toy_top_events_list, llr_nstdevs_arr4, llr_nstdevs_arr3, alpha=0.2, color='g', label=r'0.2 $\sigma_\mathrm{QCD}$ shift')
plt.fill_between(N_toy_top_events_list, llr_nstdevs_no_beta_arr4, llr_nstdevs_no_beta_arr3, alpha=0.2, color='g')
plt.fill_between(N_toy_top_events_list, llr_nstdevs_arr6, llr_nstdevs_arr5, alpha=0.2, color='b', label=r'0.3 $\sigma_\mathrm{QCD}$ shift')
plt.fill_between(N_toy_top_events_list, llr_nstdevs_no_beta_arr6, llr_nstdevs_no_beta_arr5, alpha=0.2, color='b')



#plt.plot(N_toy_top_events_list, llr_nstdevs_exact_arr2,label = r'Smeared Exact')
#plt.plot(N_toy_top_events_list, llr_nstdevs_arr3,label = r'Pois .5 cut')
plt.ylim(0,5.2)
#plt.legend(bbox_to_anchor=(0.37, 0.95))
plt.legend()
plt.xlabel(r'$N_{Top}$', fontsize=18)
plt.ylabel(r'Significance', fontsize=18)
ax.set_title(r'$P_\mathrm{cut}$(top) = %.1f' % prob_threshold, fontsize=18)



fig, ax = plt.subplots(1,1, figsize = (8,8))

llr_Pcut = '0.0'
extension0 = 'with_poisson_' + llr_Pcut + 'Pcut_1000ktoyspois_only001' # only_poisson_0Pcut_10ktoys001
llr_luminosity_arr0 = np.loadtxt(array_dir + 'testluminosityZvsNtop_arr' + extension0 + '.txt')
llr_nstdevs_arr0 = np.loadtxt(array_dir + 'testnstdevsZvsNtop_arr' + extension0 + '.txt')
llr_nstdevs_no_beta_arr0 = np.loadtxt(array_dir + 'testnstdevs_no_betaZvsNtop_arr' + extension0 + '.txt')
llr_nstdevs_exact_arr0 = np.loadtxt(array_dir + 'testnstdevs_exactZvsNtop_arr' + extension0 + '.txt')

#plt.plot(N_toy_top_events_list, llr_nstdevs_exact_arr,label = r'Not smeared Exact')
#plt.plot(N_toy_top_events_list, llr_nstdevs_arr0,label = r'%s $P_\mathrm{cut}$ Symmetric' % llr_Pcut)
plt.plot(N_toy_top_events_list, llr_nstdevs_no_beta_arr0,label = r'%s $P_\mathrm{cut}$ $\alpha=$ Asymmetric or $Z = Z(\langle \Lambda_{H_1} \rangle)$' % llr_Pcut,color='m')

ax.plot(N_toy_top_events_list, sigma_criteria1_list, label=r'%.1f $P_\mathrm{cut}$ $Z = S/\sqrt{B}$' % prob_threshold,color='g',linestyle='solid')
ax.plot(N_toy_top_events_list, sigma_criteria2_list, label=r'%.1f $P_\mathrm{cut}$ $Z = S/\sqrt{S + B}$' % prob_threshold,color='b',linestyle='dashed')
ax.plot(N_toy_top_events_list, sigma_criteria3_list, label=r'%.1f $P_\mathrm{cut}$ $Z = $ Asimov' % prob_threshold,color='y',linestyle='dotted')


plt.ylim(0,1)
plt.legend(loc='upper left')
plt.xlabel(r'$N_{top}$', fontsize=18)
plt.ylabel(r'$n_\sigma$', fontsize=18)

"""
# Z vs Pcut
luminosity = 2.0
N_top_before_cut = top_cross_section*luminosity
N_qcd_before_cut = qcd_cross_section*luminosity

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
ratio = '_0.0005ratio'
extension = '004'
llr_luminosity_arr = np.loadtxt(array_dir + 'testluminosityZvsPcut_arr' + extension + '.txt')
llr_prob_threshold_arr = np.loadtxt(array_dir + 'testprob_thresholdZvsPcut_arr' + extension + '.txt')
llr_nstdevs_arr = np.loadtxt(array_dir + 'testnstdevsZvsPcut_arr' + extension + '.txt')
llr_nstdevs_exact_arr = np.loadtxt(array_dir + 'testnstdevs_exactZvsPcut_arr' + extension + '.txt')



fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(prob_threshold_list, sigma_criteria1_list, label=r'$S/\sqrt{B}$, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_top_before_cut, N_qcd_before_cut),alpha=0.5)
ax.plot(prob_threshold_list, sigma_criteria2_list, label=r'$S/\sqrt{S + B}$, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_top_before_cut, N_qcd_before_cut),alpha=0.5)
ax.plot(prob_threshold_list, sigma_criteria3_list, label=r'Asimov, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_top_before_cut, N_qcd_before_cut),alpha=0.5)

if llr_luminosity_arr.size > 1:
    for i,luminosity in enumerate(llr_luminosity_arr):
        N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
        N_toy_mixed_events = int(luminosity*detector_efficiency*top_cross_section)
        plt.plot(llr_prob_threshold_arr, llr_nstdevs_arr[i],label = r'Approx $N_\mathrm{qcd} = %s$, $N_\mathrm{top} = %s$' % (N_toy_qcd_events, N_toy_mixed_events))
        #plt.plot(llr_prob_threshold_arr, llr_nstdevs_exact_arr[i],label = r'Exact $N_\mathrm{qcd} = %s$, $N_\mathrm{top} = %s$' % (N_toy_qcd_events, N_toy_qcd_events))
        plt.legend()
        plt.xlabel(r'$P_{cut}$')
        plt.ylabel(r'Significance $Z$')
        #plt.ylim(0,10)
else:
    N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
    N_toy_mixed_events = int(luminosity*detector_efficiency*top_cross_section)
    plt.plot(llr_prob_threshold_arr, llr_nstdevs_arr,label = r'Approx $N_\mathrm{qcd} = %s$, $N_\mathrm{top} = %s$' % (N_toy_qcd_events, N_toy_mixed_events))
    #plt.plot(llr_prob_threshold_arr, llr_nstdevs_exact_arr,label = r'$N_\mathrm{qcd} = {}, $N_\mathrm{top} = {}$'.format(N_toy_qcd_events, N_toy_qcd_events))
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
