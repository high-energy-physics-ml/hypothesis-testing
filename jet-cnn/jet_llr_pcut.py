#!/usr/bin/env python3

"""
    Script to take a pdf of prob(top) and compute the log-likelihood ratio to
    perform a hypothesis test. Outputs plot of seperation significance against
    LHC luminosity for a given probability threshold cut.
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

import seaborn as sns; sns.set(style="white", color_codes=True)

# =========================== Take in arguments ================================
import argparse

parser = argparse.ArgumentParser(description='These are the arguments that will be passed to the script')

parser.add_argument("--pcut",
                    type=float,
                    default=0,
                    help="float: The cut point of the probabilities, between 0 and 1. Default is 0.")

parser.add_argument("--ntoys",
                    type=int,
                    default=1000,
                    help="int: The number of toy events to use. Default is 1000, but use more for better accuracy.")

parser.add_argument("--ext_num",
                    type=str,
                    default=999,
                    help="str: The extension number for the output files. Should take the form of 00x, 0xy, xyz.")

parser.add_argument("--llr_terms",
                    type=str,
                    default="both",
                    help="str: LLR terms to use. 'both', 'pois', 'ml'")

args = parser.parse_args()
llr_terms = args.llr_terms

print("Pcut = " + str(args.pcut) + "ntoys = " + str(args.ntoys) + ", extension number = " + str(args.ext_num))

# =========================== Load pdf data ====================================

#top_sample=loadtxt("top.txt", unpack=True)
#qcd_sample=loadtxt("qcd.txt", unpack=True)
#top_sample = np.loadtxt("2Ccat1sProb2B.txt", unpack=True)
#qcd_sample = np.loadtxt("2Ccat2sProb2B.txt", unpack=True)
#top_sample = np.loadtxt("top.txt", unpack=True)
#qcd_sample = np.loadtxt("qcd.txt", unpack=True)

cnn_outputs = 'cnn_outputs/'

top_reference_pdf = np.loadtxt(cnn_outputs + "average_top_pdf_1000bootstraps_100bins.txt", unpack=True) # This is not the prob values but rather the pdf
qcd_reference_pdf = np.loadtxt(cnn_outputs + "average_qcd_pdf_1000bootstraps_100bins.txt", unpack=True) # This is not the prob values but rather the pdf
top_bins_centered = np.loadtxt(cnn_outputs + "top_bins_centered_1000bootstraps_100bins.txt", unpack=True)
qcd_bins_centered = np.loadtxt(cnn_outputs + "qcd_bins_centered_1000bootstraps_100bins.txt", unpack=True)

extension = 'with_poisson_' + str(args.pcut) + 'Pcut_' + str(int(args.ntoys/1000)) + 'ktoys' + str(args.ext_num)
plot_dir = 'Plots/' + extension + '/'
array_dir = 'arrays/'
fig_specification = ''
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(array_dir, exist_ok=True)
plt.close("all")
# ========================= DON'T NEED THIS WHEN ALREADY DONE IN BOOTSTRAPPING ANALYSIS CODE =================
# =========================== Find and plot pdf ================================
# Just need the histo function
def get_histo(recons_error, min_bin=0, max_bin=1, nbins=20):
    histo, bins= np.histogram(recons_error, bins=np.linspace(min_bin, max_bin, nbins), density = False)
    return histo, bins

"""
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

# ========================= DON'T NEED THIS WHEN ALREADY DONE IN BOOTSTRAPPING ANALYSIS CODE =================

# Just need the histo function

# Function to produce pdf
#def get_pdf(recons_error, min_bin=0, max_bin=1, nbins=20):
#    histo,bins = np.histogram(recons_error, bins=np.linspace(min_bin, max_bin, nbins), density = True)
#    return histo, bins


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
mixed_bins_centered = top_bins_centered

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

mixed_prob_values, mixed_reference_pdf, mixed_bins = mix_pdfs(qcd_sample, top_bins_centered, top_reference_hist, top_to_qcd_ratio)


#top_cross_section = 0.137129 # Old
#qcd_cross_section = 58.09545 # Old
top_cross_section = 53.1684
qcd_cross_section = 48829.2
top_to_qcd_ratio = top_cross_section/qcd_cross_section
#top_to_qcd_ratio = 0.0005

# This may be the better (on average) way to mix - though don't know how to get a nice pdf from it using matplotlib
mixed_reference_pdf = qcd_reference_pdf*qcd_cross_section/(qcd_cross_section + top_cross_section) + top_reference_pdf*top_cross_section/(qcd_cross_section + top_cross_section)
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
plt.savefig(plot_dir + "pdf_plot" + fig_specification + ".pdf")


# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(qcd_sample, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
ax.hist(top_sample, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.5)
ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(Top Jet)')
ax.set_title("Binary classification for QCD vs Top")
ax.set_xlim()
#plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

# New plots
# This is just a plot for visualisation purposes and is not necessary for the code
#fig, ax = plt.subplots(1,1, figsize = (8,8))
fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
ax.hist(top_sample, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.7)
ax.hist(qcd_sample, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.7)
ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'QCD + Top', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.set_xlabel(r'$P(\mathrm{Top})$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("QCD vs QCD + Top",fontsize=14)
ax.set_xlim()
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
ax.hist(top_sample, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.7)
ax.hist(qcd_sample, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.7)
ax.hist(mixed_prob_values, bins = np.linspace(0, max_bin, nbins), label = 'QCD + Top', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.set_xlabel(r'$P(\mathrm{Top})$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("QCD vs QCD + Top",fontsize=14)
ax.set_xlim()
ax.set_yscale('log')
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

# Cut PDF
cut_point = 0.5
top_sample_cut = np.stack(list(top_sample))
qcd_sample_cut = np.stack(list(qcd_sample))
mixed_sample_cut = np.stack(list(mixed_prob_values))
top_sample_cut = np.delete(top_sample_cut,np.where(top_sample_cut < cut_point)[0])
qcd_sample_cut = np.delete(qcd_sample_cut,np.where(qcd_sample_cut < cut_point)[0])
mixed_sample_cut = np.delete(mixed_sample_cut,np.where(mixed_sample_cut < cut_point)[0])

fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
ax.hist(top_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.7)
ax.hist(qcd_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.7)
ax.hist(mixed_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'QCD + Top', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.add_artist(l1)
ax.set_xlabel(r'$P(\mathrm{Top})$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("QCD vs QCD + Top",fontsize=14)
ax.set_xlim(0,1)
#ax.set_yscale('log')
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
ax.hist(top_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'Top', density = True, alpha = 0.7)
ax.hist(qcd_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.7)
ax.hist(mixed_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'QCD + Top', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.add_artist(l1)
ax.set_xlabel(r'$P(\mathrm{Top})$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("QCD vs QCD + Top",fontsize=14)
ax.set_xlim(0,1)
ax.set_yscale('log')
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")
"""

# ==============================================================================

# Instead of all this, with bootstrapping pdfs already made, just do
#top_cross_section = 0.137129 # Old
#qcd_cross_section = 58.09545 # Old
top_cross_section = 53.1684
qcd_cross_section = 48829.2
top_to_qcd_ratio = top_cross_section/qcd_cross_section
#top_to_qcd_ratio = 0.0005

mixed_reference_pdf = qcd_reference_pdf*qcd_cross_section/(qcd_cross_section + top_cross_section) + top_reference_pdf*top_cross_section/(qcd_cross_section + top_cross_section)

# ================= Define some more fucntions that will be used ===============

# Function to fit Gaussian to data
def fit_gaussian(xdata, ydata, xbins):
    # Find parameters of Gaussian; amplitude, mean, stdev
    amp = np.max(ydata)
    mu = np.mean(xdata)
    sigma = np.std(xdata)
    print(amp, mu, sigma)

    # Define the form of the Gaussian
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

    # Fit parameters for the Gaussian defined above from the data. p0 are initial guesses
    popt, _ = scipy.optimize.curve_fit(gaussian, xbins, ydata, p0 = [amp, mu, sigma])

    # Now get the Gaussian curve with those fitted parameters
    fitted_gaussian = gaussian(xbins, *popt)
    return fitted_gaussian, popt

# Function to run over pdfs and compute sum log likelihoods to obtain alpha
def get_alpha(LLRqcd_list, LLRtop_list, LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins):
    # Recenter bins
    LLRqcd_binscenters = np.array([0.5 * (LLRqcd_bins[i] + LLRqcd_bins[i+1]) for i in range(len(LLRqcd_bins)-1)])
    LLRtop_binscenters = np.array([0.5 * (LLRtop_bins[i] + LLRtop_bins[i+1]) for i in range(len(LLRtop_bins)-1)])

    # Get parameters for Gaussian fit
    _, test_gaus_params = fit_gaussian(LLRqcd_list, LLRqcd_histo, LLRqcd_binscenters)
    _, anomaly_gaus_params = fit_gaussian(LLRtop_list, LLRtop_histo, LLRtop_binscenters)

    test_amplitude = test_gaus_params[0]
    test_mean = test_gaus_params[1]
    test_stdev = abs(test_gaus_params[2]) # abs since for some weird numerical reason stdev can be -ve. (it does not matter since it is squared but it makes me feel uncomfortable unless it is +ve)

    anomaly_amplitude = anomaly_gaus_params[0]
    anomaly_mean = anomaly_gaus_params[1]
    anomaly_stdev = abs(anomaly_gaus_params[2])

    # Integrate over Gaussian distribution with different lambda_cut as limits
    # to get alpha for which alpha = beta
    def integrand(x, amplitude, mean, stdev):
        return amplitude * np.exp(-((x - mean) / 4 / stdev)**2)


    #lam_cut_potential_values = np.linspace(-200, -80, 1000)

    # min and max lamda values (would be infinity but scipy.integrate.quad breaks when using scipy.inf)
    print("====================== VALUES ================================")
    print(test_amplitude, test_mean, test_stdev, anomaly_amplitude, anomaly_mean, anomaly_stdev)
    min_lam = anomaly_mean - 50*np.average((test_stdev, anomaly_stdev))
    max_lam = test_mean + 50*np.average((test_stdev, anomaly_stdev))
    print(min_lam, max_lam)

    lam_cut_potential_values = np.linspace(anomaly_mean - 20*np.average((test_stdev, anomaly_stdev)), test_mean + 20*np.average((test_stdev, anomaly_stdev)), 100000)

    # Alpha and beta normalisations
    alpha_normalisation ,alpha_normalisation_error = integrate.quad(integrand,min_lam,max_lam,args=(test_amplitude, test_mean, test_stdev))
    beta_normalisation ,beta_normalisation_error = integrate.quad(integrand,min_lam,max_lam,args=(anomaly_amplitude, anomaly_mean, anomaly_stdev))
    print("========NORMALISATION:======================")
    print("HELLO")
    print(alpha_normalisation, beta_normalisation)

    # Integrate from increasing values of lam_cut to find alpha which is closest to beta
    alpha_list = []
    beta_list = []
    alpha_beta_diff_list = []
    for i,lam_cut in enumerate(lam_cut_potential_values):
        alpha_integral1 ,alpha_integral1_error = integrate.quad(integrand,min_lam,lam_cut,args=(test_amplitude, test_mean, test_stdev))
        alpha = alpha_integral1/alpha_normalisation
        alpha_list.append(alpha)

        beta_integral1 ,beta_integral1_error = integrate.quad(integrand,lam_cut,max_lam,args=(anomaly_amplitude, anomaly_mean, anomaly_stdev))
        beta = beta_integral1/beta_normalisation

        # Create a list of the difference between alpha and beta - we will search for the value that is closest to zero
        alpha_beta_diff = abs(alpha - beta)
        alpha_beta_diff_list.append(alpha_beta_diff)
        beta_list.append(beta)
        #print(i, lam_cut, alpha_integral1, alpha, beta_integral1, beta, alpha_beta_diff)

    # Get the value of lam_cut for which alpha is closest to beta (approx of alpha = beta)
    closest_index = alpha_beta_diff_list.index(min(alpha_beta_diff_list))
    lam_cut = lam_cut_potential_values[closest_index]

    # Get the value of alpha for which alpha is closest to beta (approx of alpha = beta)
    alpha = alpha_list[closest_index]

    return alpha

def get_alpha_no_beta(LLRqcd_list, LLRtop_list, LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins):
    # Recenter bins
    LLRqcd_binscenters = np.array([0.5 * (LLRqcd_bins[i] + LLRqcd_bins[i+1]) for i in range(len(LLRqcd_bins)-1)])
    LLRtop_binscenters = np.array([0.5 * (LLRtop_bins[i] + LLRtop_bins[i+1]) for i in range(len(LLRtop_bins)-1)])

    # Get parameters for Gaussian fit
    _, test_gaus_params = fit_gaussian(LLRqcd_list, LLRqcd_histo, LLRqcd_binscenters)
    _, anomaly_gaus_params = fit_gaussian(LLRtop_list, LLRtop_histo, LLRtop_binscenters)

    test_amplitude = test_gaus_params[0]
    test_mean = test_gaus_params[1]
    test_stdev = abs(test_gaus_params[2]) # abs since for some weird numerical reason stdev can be -ve. (it does not matter since it is squared but it makes me feel uncomfortable unless it is +ve)

    anomaly_amplitude = anomaly_gaus_params[0]
    anomaly_mean = anomaly_gaus_params[1]
    anomaly_stdev = abs(anomaly_gaus_params[2])

    # Integrate over Gaussian distribution with different lambda_cut as limits
    # to get alpha for which alpha = beta
    def integrand(x, amplitude, mean, stdev):
        return amplitude * np.exp(-((x - mean) / 4 / stdev)**2)


    #lam_cut_potential_values = np.linspace(-200, -80, 1000)

    # min and max lamda values (would be infinity but scipy.integrate.quad breaks when using scipy.inf)
    print("====================== VALUES ================================")
    print(test_amplitude, test_mean, test_stdev, anomaly_amplitude, anomaly_mean, anomaly_stdev)
    min_lam = anomaly_mean - 50*np.average((test_stdev, anomaly_stdev))
    max_lam = test_mean + 50*np.average((test_stdev, anomaly_stdev))
    print(min_lam, max_lam)

    lam_cut_potential_values = np.linspace(anomaly_mean - 20*np.average((test_stdev, anomaly_stdev)), test_mean + 20*np.average((test_stdev, anomaly_stdev)), 100000)

    # Alpha and beta normalisations
    alpha_normalisation ,alpha_normalisation_error = integrate.quad(integrand,min_lam,max_lam,args=(test_amplitude, test_mean, test_stdev))
    beta_normalisation ,beta_normalisation_error = integrate.quad(integrand,min_lam,max_lam,args=(anomaly_amplitude, anomaly_mean, anomaly_stdev))
    print("========NORMALISATION:======================")
    print("HELLO")
    print(alpha_normalisation, beta_normalisation)

    # Now take lam cut to be mean of mixed LLR
    lam_cut = np.mean(LLRtop_list)
    print("LAM CUT",lam_cut)
    alpha_integral1 ,alpha_integral1_error = integrate.quad(integrand,min_lam,lam_cut,args=(test_amplitude, test_mean, test_stdev))
    alpha = alpha_integral1/alpha_normalisation

    return alpha

def get_alpha_exact(LLRqcd_list, LLRtop_list, LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins):
    # Loop through all bins, calculating the area under LLR hist A and B and find the bin where areas are most equal
    #LLRA_values, LLRA_bins,_ = plt.hist(LLRqcd_list, bins = np.linspace(min_llr, max_llr, 100), label = 'QCD', alpha = 0.5)
    #LLRB_values, LLRB_bins,_ = plt.hist(LLRtop_list, bins = np.linspace(min_llr, max_llr, 100), label = 'Top', alpha = 0.5)

    LLRA_values,LLRA_bins = LLRqcd_histo, LLRqcd_bins
    LLRB_values,LLRB_bins = LLRtop_histo, LLRtop_bins

    alpha_list = []
    beta_list = []
    alpha_beta_diff_list = []
    # A and B share the same bins
    # A is to the left of B
    for i, bin in enumerate(LLRA_bins):
        alpha = sum(np.diff(LLRA_bins[i:])*LLRA_values[i:])/sum(np.diff(LLRA_bins)*LLRA_values)
        beta = sum(np.diff(LLRB_bins[:(i+2)])*LLRB_values[:(i+1)])/sum(np.diff(LLRB_bins)*LLRB_values)
        alpha_list.append(alpha)
        beta_list.append(beta)

        alpha_beta_diff = abs(alpha - beta)
        alpha_beta_diff_list.append(alpha_beta_diff)

    # Get the value of lam_cut for which alpha is closest to beta (approx of alpha = beta)
    closest_index = alpha_beta_diff_list.index(min(alpha_beta_diff_list))
    bin_cut = LLRA_bins[closest_index]

    # Get the value of alpha for which alpha is closest to beta (approx of alpha = beta)
    alpha = alpha_list[closest_index]

    return alpha


# Function to get standard deviations of seperation from alpha:
# For the found value of alpha, find the corresponding number of standard
# deviations n by solving
# alpha = (1/sqrt(2 pi)) * int_n^inf exp(-x^2/2) dx
# Note that this is the one-sided defintion and i don't think that it is well
# defined for alpha < 0.5, though the correct result could be obtained for alpha > 0.5
# if you do alpha -> 1 - alpha
def get_nstdevs(alpha):
    # The integrand
    def integrand(x):
        return np.exp(-x**2/2)

    # The function to solve for - this is the above equation rearranged
    # Max number of standard deviation to integrate up to (a number -> infity)
    # For some reason n ~> 10000 gives bad integration but 1000 is more than enough
    n_max = 1000
    def func(n):
        integral,error = integrate.quad(integrand, n, n_max)
        return integral - alpha*np.sqrt(2.0*np.pi)

    # Solve
    n_estimate = 1.0
    sol = scipy.optimize.fsolve(func, n_estimate)

    return sol[0]

# ============== Define the log-likelihood sampling procedure===================

# Function to sample from toy experiments and return the log likelihoods for qcd
# and top if they were to be sampled from either top or qcd
def sample_ll_from_toys(pdfA, pdfA_bins, pdfB, N_toys=10000, N_toy_events=50):
    # For each toy experiment we will find the LLR
    toy_log_likelihood_sum_listA = []
    toy_log_likelihood_sum_listB = []

    for toy in range(N_toys):
        # Get sample bin values for pdf A, using pdf A as weights

        toy_histonum = random.choices(pdfA_bins, weights=pdfA, k=N_toy_events[toy])
        #print(N_toy_events[toy])

        # Get the histogram of events corresponding to the randomly sampled bin/x-axis values
        # nbins must be the same as nbins of pdf
        nbins = len(pdfA) + 1
        min_P = np.min(pdfA_bins)
        #max_P = 1
        max_P = np.max(pdfA_bins) + np.diff(pdfA_bins)[0]/2 # Not sure why I have to do this to get the plot good, but I do
        toy_histo, _ = get_histo(toy_histonum, min_P, max_P, nbins)

        # Find the the likelihood for each toy by taking the product of all pdfs and take the log
        # Here I actually sum the log likelihood but when the product/sum is performed is irrelevant
        toy_log_likelihood_sumA = 0.0
        toy_log_likelihood_sumB = 0.0

        # For each event in the toy sample:
        for i in range(len(toy_histo)):
            if pdfA[i] > 0:
                # Find the likelihood and log-likelihood that one would get from the reference histogram
                LHA = pdfA[i] # this is not corresponding to the right value in the sample histo - well it is as long as nbins of reference histo and toy histo are the same
                LLA = -2*log(LHA)
                # Multiply the log-likelihood from the reference histogram by the number of samples in the corresponding bin, works since ll is a sum
                toy_log_likelihoodA = toy_histo[i]*LLA
                toy_log_likelihood_sumA += toy_log_likelihoodA
        toy_log_likelihood_sum_listA.append(toy_log_likelihood_sumA)

        for i in range(len(toy_histo)):
            if pdfB[i] > 0:
                # Find the likelihood and log-likelihood that one would get from the reference histogram
                LHB = pdfB[i] # this is not corresponding to the right value in the sample histo - well it is as long as nbins of reference histo and toy histo are the same
                LLB = -2*log(LHB)
                # Multiply the log-likelihood from the reference histogram by the number of samples in the corresponding bin, works since ll is a sum
                toy_log_likelihoodB = toy_histo[i]*LLB
                toy_log_likelihood_sumB += toy_log_likelihoodB
        toy_log_likelihood_sum_listB.append(toy_log_likelihood_sumB)
    print("MIN P", min_P)
    print("MAX P", max_P, np.diff(pdfA_bins)[0], nbins)
    return toy_log_likelihood_sum_listA, toy_log_likelihood_sum_listB

# ======================== Define the fucntion that will plot the log likelihood ratios ======================

def plot_llr(LLRqcd, LLRtop, anomaly_type, N_toys, N_toy_events, prob_threshold):

    # Plot and get histogram of -2ln(lambda) sampled from each toy experiment
    fig = plt.figure(figsize = (6,6))
    ax=fig.add_axes([0.13,0.14,0.8,0.8])
    min_llr = np.floor(min(np.min(LLRqcd), np.min(LLRtop)))
    max_llr = np.ceil(max(np.max(LLRqcd), np.max(LLRtop)))
    nbins = 100
    LLRtop_histo, LLRtop_bins, _ = ax.hist(LLRtop, bins=np.linspace(min_llr,max_llr,100),label='QCD + TOP', alpha = 0.7)
    LLRqcd_histo, LLRqcd_bins, _ = ax.hist(LLRqcd, bins=np.linspace(min_llr,max_llr,100),label='QCD',alpha = 0.7)
    # This requires centering the bins so that we can accurately fit a Gaussian
    LLRqcd_binscenters = np.array([0.5 * (LLRqcd_bins[i] + LLRqcd_bins[i+1]) for i in range(len(LLRqcd_bins)-1)])
    LLRtop_binscenters = np.array([0.5 * (LLRtop_bins[i] + LLRtop_bins[i+1]) for i in range(len(LLRtop_bins)-1)])

    #print("LLRqcd_binscenters",LLRqcd_binscenters)
    #print("LLRqcd_histo",LLRqcd_histo)

    # Fit a Gaussian
    LLRqcd_gaus, _ = fit_gaussian(LLRqcd, LLRqcd_histo, LLRqcd_binscenters)
    LLRtop_gaus, _ = fit_gaussian(LLRtop, LLRtop_histo, LLRtop_binscenters)

    # Plot the Gaussian
    ax.plot(LLRqcd_binscenters, LLRqcd_gaus, 'C1')
    ax.plot(LLRtop_binscenters, LLRtop_gaus, 'C0')
    ax.set_title(r'QCD vs QCD + Top, $P_\mathrm{cut}(top)$ = %s' % prob_threshold, fontsize=14)
    #ax.set_title(r'QCD vs QCD + Top', fontsize=14)
    ax.set_xlabel(r'LLR')
    ax.set_ylabel(r'Frequency')
    l1=ax.legend(loc=1,fontsize=14)
    l2=ax.legend([r"$N_\mathrm{qcd \: events} = %s$" "\n" "$N_\mathrm{top \: events} = %s$" % (N_toy_events[0],N_toy_events[1]-N_toy_events[0])],loc=2,prop={'size':14},handlelength=0,handletextpad=0)
    ax.add_artist(l1)
    ax.set_xlabel('LLR',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.savefig(plot_dir + "QCD{}_LLR_{}toy_events".format(anomaly_type, N_toy_events) + fig_specification + ".pdf")

    return LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins

# ================================== Some binning stuff ======================================================

# A check that area of pdf = 1:
qcd_bin_width = qcd_bins_centered[1] - qcd_bins_centered[0]
top_bin_width = top_bins_centered[1] - top_bins_centered[0]
print("Area of unscaled qcd pdf =",qcd_reference_pdf.sum()*qcd_bin_width)
print("Area of unscaled top pdf =",top_reference_pdf.sum()*top_bin_width)

mixed_bins_centered = qcd_bins_centered
mixed_bin_width = qcd_bin_width

# =================================== Plot -2nln(lambda) for an individual toy ==================================
# I have removed this to keep the code concise but if we want to add it we can adapt it from an older version


# = Run the scipt to get LLR distributions, and calcualte seperation for a number of toy experiments ======================

def run_toys_luminosity(luminosity, prob_threshold, qcd_cross_section, top_cross_section, detector_efficiency):
    # Get total number of events - these will be used as means in the Poisson distributions
    mean_N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
    mean_N_toy_mixed_events = int(luminosity*detector_efficiency*(qcd_cross_section + top_cross_section)) # N will be greater for mixed? Well this is the proper way of doing it - luminosity is the thing controlled, not number of events

    # Get n events for each toy as a number sampled from the poisson distribution with some mean
    # This list will be used for n events in both poisson factor and probs pdf factor
    N_toy_qcd_events_list = []
    N_toy_mixed_events_list = []
    for toy in range(N_toys):
        N_toy_qcd_events = np.random.poisson(mean_N_toy_qcd_events)
        N_toy_mixed_events = np.random.poisson(mean_N_toy_mixed_events)
        N_toy_qcd_events_list.append(N_toy_qcd_events)
        N_toy_mixed_events_list.append(N_toy_mixed_events)
    #print(N_toy_qcd_events_list)
    #print(N_toy_mixed_events_list)


    print("\n\n")
    print("Pcut:",prob_threshold)
    print("luminosity:", luminosity)
    print("N QCD toys",mean_N_toy_qcd_events)
    print("N mixed toys",mean_N_toy_mixed_events)

    # Make a cut on the pdf
    print("fraction of QCD pdf before:",qcd_reference_pdf.sum()*qcd_bin_width)
    print("fraction of mixed pdf before cut:",mixed_reference_pdf.sum()*top_bin_width)

    # Copy and cut on pdf and bins
    qcd_reference_pdf_cut = np.stack(list(qcd_reference_pdf))
    mixed_reference_pdf_cut = np.stack(list(mixed_reference_pdf))
    qcd_bins_centered_cut = np.stack(list(qcd_bins_centered))
    mixed_bins_centered_cut = np.stack(list(mixed_bins_centered))

    qcd_reference_pdf_cut = np.delete(qcd_reference_pdf_cut,np.where(qcd_bins_centered < prob_threshold)[0])
    mixed_reference_pdf_cut = np.delete(mixed_reference_pdf_cut,np.where(mixed_bins_centered < prob_threshold)[0])
    qcd_bins_centered_cut = np.delete(qcd_bins_centered_cut,np.where(qcd_bins_centered < prob_threshold)[0])
    mixed_bins_centered_cut = np.delete(mixed_bins_centered_cut,np.where(mixed_bins_centered < prob_threshold)[0])

    # Reduce number of events if cutting
    epsilon_qcd = qcd_reference_pdf_cut.sum()*qcd_bin_width
    epsilon_mixed = mixed_reference_pdf_cut.sum()*mixed_bin_width

    mean_N_qcd_after_cut = int(epsilon_qcd*mean_N_toy_qcd_events)
    mean_N_mixed_after_cut = int(epsilon_mixed*mean_N_toy_mixed_events)

    N_qcd_after_cut_list = []
    N_mixed_after_cut_list = []
    for toy in range(N_toys):
        N_qcd_after_cut = int(epsilon_qcd*N_toy_qcd_events_list[toy])
        N_mixed_after_cut = int(epsilon_mixed*N_toy_mixed_events_list[toy])
        N_qcd_after_cut_list.append(N_qcd_after_cut)
        N_mixed_after_cut_list.append(N_mixed_after_cut)

        #print("N QCD before cut:", N_toy_qcd_events_list[toy], "N QCD after cut:", N_qcd_after_cut, "fraction of QCD pdf remaining after cut:",qcd_reference_pdf_cut.sum()*qcd_bin_width) # this will be wrong now, but it is just a print
        #print("N mixed before cut:", N_toy_mixed_events_list[toy], "N mixed after cut:", N_mixed_after_cut, "fraction of mixed pdf remaining after cut:",mixed_reference_pdf_cut.sum()*mixed_bin_width)# this will be wrong now, but it is just a print

    print("The qcd bins that will be sampled from:",qcd_bins_centered_cut)
    print("The mixed bins that will be sampled from:",mixed_bins_centered_cut)
    print("The qcd pdfs that will be sampled from:",qcd_reference_pdf_cut)
    print("The mixed pdfs that will be sampled from:",mixed_reference_pdf_cut)

    # Renorm pdfs
    qcd_reference_pdf_cut = qcd_reference_pdf_cut*(1.0/(qcd_reference_pdf_cut.sum()*np.diff(qcd_bins_centered_cut)[0]))
    mixed_reference_pdf_cut = mixed_reference_pdf_cut*(1.0/(mixed_reference_pdf_cut.sum()*np.diff(mixed_bins_centered_cut)[0]))
    print("PDF RENORMALISATION CHECK",qcd_reference_pdf_cut.sum()*qcd_bin_width)
    print("PDF RENORMALISATION CHECK",mixed_reference_pdf_cut.sum()*mixed_bin_width)

    # Plot PDFs after making a cut - this method is bad because matplotlib bar sucks for plotting
    plot_cut_pdfs = False
    if plot_cut_pdfs == True:
        fig, ax = plt.subplots(1,1, figsize = (8,8))
        #ax.hist(qcd_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'QCD', density = True, alpha = 0.5)
        #ax.hist(mixed_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
        ax.bar(mixed_bins_centered_cut, mixed_reference_pdf_cut, width=np.diff(mixed_bins_centered_cut)[0], label = 'QCD+Top', alpha = 0.7)
        ax.bar(qcd_bins_centered_cut, qcd_reference_pdf_cut, width=np.diff(qcd_bins_centered_cut)[0], label = 'QCD', alpha = 0.7)
        ax.legend()
        ax.set_xlabel(r'$P(\mathrm{Top})$')
        ax.set_title(r"QCD vs Top $P_\mathrm{cut}(\mathrm{Top}) = %s$" % prob_threshold)
        ax.set_xlim()
        plt.savefig(plot_dir + "pdfs_with_pcut" + str(prob_threshold) + 'Pcut' + fig_specification + ".pdf")


    # Sample
    cut_probs_pdf = False
    # If cutting on probs PDF
    if cut_probs_pdf == True:
        qcd_sample_toy_log_likelihoodqcd, qcd_sample_toy_log_likelihoodtop = sample_ll_from_toys(qcd_reference_pdf_cut,qcd_bins_centered_cut,mixed_reference_pdf_cut, N_toys = N_toys, N_toy_events = N_qcd_after_cut_list)
        top_sample_toy_log_likelihoodtop, top_sample_toy_log_likelihoodqcd = sample_ll_from_toys(mixed_reference_pdf_cut,mixed_bins_centered_cut, qcd_reference_pdf_cut, N_toys = N_toys, N_toy_events = N_mixed_after_cut_list)

    # If not cutting on probs PDF
    elif cut_probs_pdf != True:
        qcd_sample_toy_log_likelihoodqcd, qcd_sample_toy_log_likelihoodtop = sample_ll_from_toys(qcd_reference_pdf,qcd_bins_centered,mixed_reference_pdf, N_toys = N_toys, N_toy_events = N_toy_qcd_events_list)
        top_sample_toy_log_likelihoodtop, top_sample_toy_log_likelihoodqcd = sample_ll_from_toys(mixed_reference_pdf,mixed_bins_centered, qcd_reference_pdf, N_toys = N_toys, N_toy_events = N_toy_mixed_events_list)

    print("fraction of QCD pdf remaining after cut:",qcd_reference_pdf_cut.sum()*qcd_bin_width)
    print("fraction of mixed pdf remaining after cut:",mixed_reference_pdf_cut.sum()*top_bin_width)

    # Calculate Poisson factor
    poisson_events_from_probs_cut = True
    if poisson_events_from_probs_cut == True:
        n_qcd_list = N_qcd_after_cut_list
        n_mixed_list = N_mixed_after_cut_list
        mu_qcd = mean_N_qcd_after_cut
        mu_mixed = mean_N_mixed_after_cut
    elif poisson_events_from_probs_cut != True:
        n_qcd_list = N_toy_qcd_events_list
        n_mixed_list = N_toy_mixed_events_list
        mu_qcd = mean_N_toy_qcd_events
        mu_mixed = mean_N_toy_mixed_events

    LLR_qcd_poisson_list = []
    LLR_mixed_poisson_list = []
    for toy in range(N_toys):
        LLR_qcd_poisson = -2*(n_qcd_list[toy]*log(mu_qcd/mu_mixed) + (mu_mixed - mu_qcd))
        LLR_mixed_poisson = -2*(n_mixed_list[toy]*log(mu_qcd/mu_mixed) + (mu_mixed - mu_qcd))
        #print("LLR QCD Poisson factor:", LLR_qcd_poisson)
        #print("LLR mixed Poisson factor:", LLR_mixed_poisson)
        LLR_qcd_poisson_list.append(LLR_qcd_poisson)
        LLR_mixed_poisson_list.append(LLR_mixed_poisson)

    #print("LLR QCD Poisson factor:", LLR_qcd_poisson)
    #print("LLR mixed Poisson factor:", LLR_mixed_poisson)

    # Calculate ratio
    LLRqcd_list=[]
    LLRtop_list=[]
    for i in range(N_toys):
        if llr_terms == "both":
            LLRqcd = LLR_qcd_poisson_list[i] + qcd_sample_toy_log_likelihoodqcd[i]-qcd_sample_toy_log_likelihoodtop[i]
            LLRtop = LLR_mixed_poisson_list[i] + top_sample_toy_log_likelihoodqcd[i]-top_sample_toy_log_likelihoodtop[i]
        elif llr_terms == "pois":
            LLRqcd = LLR_qcd_poisson_list[i]
            LLRtop = LLR_mixed_poisson_list[i]
        elif llr_terms == "ml":
            LLRqcd = qcd_sample_toy_log_likelihoodqcd[i]-qcd_sample_toy_log_likelihoodtop[i]
            LLRtop = top_sample_toy_log_likelihoodqcd[i]-top_sample_toy_log_likelihoodtop[i]

        # Test with only Poisson factor
        #LLRqcd = LLR_qcd_poisson_list[i]
        #LLRtop = LLR_mixed_poisson_list[i]

        #print("\n",LLR_qcd_poisson_list[i], qcd_sample_toy_log_likelihoodqcd[i]-qcd_sample_toy_log_likelihoodtop[i], LLRqcd)
        #print(LLR_mixed_poisson_list[i], top_sample_toy_log_likelihoodqcd[i]-top_sample_toy_log_likelihoodtop[i], LLRtop)
        #print("\n",LLR_qcd_poisson_list[i])
        #print(LLR_mixed_poisson_list[i])
        LLRqcd_list.append(LLRqcd)
        LLRtop_list.append(LLRtop)

    # Plot
    LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins= plot_llr(LLRqcd_list, LLRtop_list, 'top', N_toys = 10000, N_toy_events = (mean_N_toy_qcd_events, mean_N_toy_mixed_events), prob_threshold=prob_threshold)

    # Calculate alpha and n standard deviations
    import math
    try:
        alpha = get_alpha(LLRqcd_list, LLRtop_list, LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins)
        alpha = 1.0 - alpha
        if alpha > 0.5: # Shouldn't be needed but will keep in case
            alpha = 1.0 - alpha
        if math.isnan(alpha):
            alpha = 0.5
        if mean_N_toy_qcd_events == mean_N_toy_mixed_events:
            alpha = 0.5
    except:
        alpha = 0.5
    try:
        alpha_no_beta = get_alpha_no_beta(LLRqcd_list, LLRtop_list, LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins)
        alpha_no_beta = 1.0 - alpha_no_beta
        if alpha_no_beta > 0.5: # Shouldn't be needed but will keep in case
            alpha_no_beta = 1.0 - alpha
        if math.isnan(alpha_no_beta):
            alpha_no_beta = 0.5
        if mean_N_toy_qcd_events == mean_N_toy_mixed_events:
            alpha_no_beta = 0.5
    except:
        alpha_no_beta = 0.5
    try:
        alpha_exact = get_alpha_exact(LLRqcd_list, LLRtop_list, LLRqcd_histo, LLRqcd_bins, LLRtop_histo, LLRtop_bins)
        if alpha_exact > 0.5: # Shouldn't be needed but will keep in case
            alpha_exact = 1.0 - alpha_exact
        if math.isnan(alpha_exact):
            alpha_exact = 0.5
        if mean_N_toy_qcd_events == mean_N_toy_mixed_events:
            alpha_exact = 0.5
    except:
        alpha_exact = 0.5

    nstdevs = get_nstdevs(alpha)
    nstdevs_no_beta = get_nstdevs(alpha_no_beta)
    nstdevs_exact = get_nstdevs(alpha_exact)
    print("alpha:",alpha, "nstdevs:", nstdevs)
    print("alpha no beta:",alpha_no_beta, "nstdevs no beta:", nstdevs_no_beta)
    print("alpha exact", alpha_exact, "nstdevs exact:", nstdevs_exact)

    return alpha, nstdevs, alpha_no_beta, nstdevs_no_beta, alpha_exact, nstdevs_exact

# ========================== Z vs Ntop ===========================================

# During development, we may set a cross section ratio instead of using the actual cross sections. So get the top cross section here
#top_cross_section = top_to_qcd_ratio*qcd_cross_section

# Convert cross sections from pb to fb
#qcd_cross_section = qcd_cross_section*10**3 # Already done
#top_cross_section = top_cross_section*10**3

detector_efficiency = 1
#N_toys = 100000
N_toys = args.ntoys

"""
prob_threshold = args.pcut
luminosity_arr = np.linspace(0.01,10,11)
nstdevs_list = []
nstdevs_no_beta_list = []
nstdevs_exact_list = []
for luminosity in luminosity_arr:
    alpha, nstdevs, alpha_no_beta, nstdevs_no_beta, alpha_exact, nstdevs_exact = run_toys_luminosity(luminosity, prob_threshold, qcd_cross_section, top_cross_section, detector_efficiency)
    nstdevs_list.append(nstdevs)
    nstdevs_no_beta_list.append(nstdevs_no_beta)
    nstdevs_exact_list.append(nstdevs_exact)


# Save the arrays so we can plot them later

nstdevs_arr = np.stack((nstdevs_list))
nstdevs_no_beta_arr = np.stack((nstdevs_no_beta_list))
nstdevs_exact_arr = np.stack((nstdevs_exact_list))
np.savetxt(array_dir + 'testluminosityZvsNtop_arr' + extension + '.txt', luminosity_arr)
np.savetxt(array_dir + 'testnstdevsZvsNtop_arr' + extension + '.txt', nstdevs_arr)
np.savetxt(array_dir + 'testnstdevs_no_betaZvsNtop_arr' + extension + '.txt', nstdevs_no_beta_arr)
np.savetxt(array_dir + 'testnstdevs_exactZvsNtop_arr' + extension + '.txt', nstdevs_exact_arr)

# Load them in again
luminosity_arr = np.loadtxt(array_dir + 'testluminosityZvsNtop_arr' + extension + '.txt')
nstdevs_arr = np.loadtxt(array_dir + 'testnstdevsZvsNtop_arr' + extension + '.txt')
nstdevs_no_beta_arr = np.loadtxt(array_dir + 'testnstdevs_no_betaZvsNtop_arr' + extension + '.txt')
nstdevs_exact_arr = np.loadtxt(array_dir + 'testnstdevs_exactZvsNtop_arr' + extension + '.txt')
"""
"""
plt.figure()
#N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
N_toy_mixed_events = int(luminosity*detector_efficiency*top_cross_section)
plt.plot(luminosity_arr, nstdevs_arr,label = r'Approx')
plt.plot(luminosity_arr, nstdevs_exact_arr,label = r'Exact')
plt.legend()
plt.xlabel(r'$L$')
plt.ylabel(r'Significance $Z$')

plt.figure()
#N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
#N_toy_mixed_events = int(luminosity*detector_efficiency*top_cross_section)
N_toy_qcd_events_list = []
N_toy_top_events_list = []
for luminosity in luminosity_arr:
    #N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
    N_toy_top_events = int(luminosity*detector_efficiency*top_cross_section)
    #N_toy_qcd_events_list.append(N_toy_qcd_events)
    N_toy_top_events_list.append(N_toy_top_events)
plt.plot(N_toy_top_events_list, nstdevs_arr,label = r'Approx $N_\mathrm{SM} = $, $N_\mathrm{EFT} = $')
plt.plot(N_toy_top_events_list, nstdevs_exact_arr,label = r'Exact $N_\mathrm{SM} =s$, $N_\mathrm{EFT} = $')
plt.legend()
plt.xlabel(r'$N_{top}$')
plt.ylabel(r'Significance $Z$')

"""

# =========================== Z vs Pcut ===========================================

luminosity_arr = np.linspace(5,5,1)
#prob_threshold_list = [0.2,0.4,0.6,0.8]
prob_threshold_arr = np.linspace(0,0.8,4)

plt.figure()
for luminosity in luminosity_arr:
    alpha_list = []
    nstdevs_list = []
    nstdevs_no_beta_list = []
    alpha_exact_list = []
    nstdevs_exact_list = []
    for prob_threshold in prob_threshold_arr:
        alpha, nstdevs, alpha_no_beta, nstdevs_no_beta, alpha_exact, nstdevs_exact = run_toys_luminosity(luminosity, prob_threshold, qcd_cross_section, top_cross_section, detector_efficiency)
        nstdevs_list.append(nstdevs)
        nstdevs_no_beta_list.append(nstdevs_no_beta)
        nstdevs_exact_list.append(nstdevs_exact)



# Save the arrays so we can plot them later
nstdevs_arr = np.stack((nstdevs_list))
nstdevs_no_beta_arr = np.stack((nstdevs_no_beta_list))
nstdevs_exact_arr = np.stack((nstdevs_exact_list))
np.savetxt(array_dir + 'luminosityZvsPcut_arr' + extension + '.txt', luminosity_arr)
np.savetxt(array_dir + 'prob_thresholdZvsPcut_arr' + extension + '.txt', prob_threshold_arr)
np.savetxt(array_dir + 'nstdevsZvsPcut_arr' + extension + '.txt', nstdevs_arr)
np.savetxt(array_dir + 'nstdevs_no_betaZvsPcut_arr' + extension + '.txt', nstdevs_no_beta_arr)
np.savetxt(array_dir + 'nstdevs_exactZvsPcut_arr' + extension + '.txt', nstdevs_exact_arr)

# Load them in again
luminosity_arr = np.loadtxt(array_dir + 'luminosityZvsPcut_arr' + extension + '.txt')
prob_threshold_arr = np.loadtxt(array_dir + 'prob_thresholdZvsPcut_arr' + extension + '.txt')
nstdevs_arr = np.loadtxt(array_dir + 'nstdevsZvsPcut_arr' + extension + '.txt')
nstdevs_no_beta_arr = np.loadtxt(array_dir + 'nstdevs_no_betaZvsPcut_arr' + extension + '.txt')
nstdevs_exact_arr = np.loadtxt(array_dir + 'nstdevs_exactZvsPcut_arr' + extension + '.txt')

plt.figure()
if luminosity_arr.size > 1:
    for i,luminosity in enumerate(luminosity_arr):
        N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
        N_toy_mixed_events = int(luminosity*detector_efficiency*top_cross_section)
        plt.plot(prob_threshold_arr, nstdevs_arr[i],label = r'Approx $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_qcd_events, N_toy_qcd_events))
        plt.plot(prob_threshold_arr, nstdevs_exact_arr[i],label = r'Exact $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_qcd_events, N_toy_qcd_events))
        plt.legend()
        plt.xlabel(r'$P_{cut}$')
        plt.ylabel(r'Significance $Z$')
        #plt.ylim(0,10)
else:
    N_toy_qcd_events = int(luminosity*detector_efficiency*qcd_cross_section)
    N_toy_mixed_events = int(luminosity*detector_efficiency*top_cross_section)
    plt.plot(prob_threshold_arr, nstdevs_arr,label = r'Symm $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_qcd_events, N_toy_qcd_events))
    plt.plot(prob_threshold_arr, nstdevs_no_beta_arr,label = r'Asym$N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_qcd_events, N_toy_qcd_events))
    #plt.plot(prob_threshold_arr, nstdevs_arr,label = r'Exact $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_qcd_events, N_toy_qcd_events))
    plt.legend()
    plt.xlabel(r'$P_{cut}$')
    plt.ylabel(r'Significance $Z$')
    #plt.ylim(0,10)


# Run for specific values only for LLR plotting purposes
#luminosity = 2.0
#prob_threshold = 0
#alpha, nstdevs, alpha_exact, nstdevs_exact = run_toys_luminosity(luminosity, prob_threshold, qcd_cross_section, top_cross_section, detector_efficiency)

#plt.show(block=False)
plt.ion()
plt.show()
