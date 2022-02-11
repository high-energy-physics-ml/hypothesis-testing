#!/usr/bin/env python3

"""
    Script to take a pdf of prob(eft) and compute the log-likelihood ratio to
    perform a hypothesis test. Outputs plot of seperation significance against
    LHC luminosity for a given probability threshold cut.
    __author__ = "Michael Soughton", "Charanjit Kaur Khosa", "Veronica Sanz"
    __email__ =
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from keras.layers import Input, Dense, Lambda, Flatten, Reshape
#from keras.models import Model
#from keras import backend as K
#from keras import metrics
import scipy.optimize
from scipy.stats import norm
from scipy import integrate
import os
import random
from numpy import log

import seaborn as sns; sns.set(style="white", color_codes=True)

plt.close("all")

# =========================== Take in arguments ================================
import argparse

parser = argparse.ArgumentParser(description='These are the arguments that will be passed to the script')

parser.add_argument("--rcut",
                    type=float,
                    default=0,
                    help="float: The cut point of the probabilities, between 0 and x. Default is 0.")

parser.add_argument("--ntoys",
                    type=int,
                    default=1000,
                    help="int: The number of toy events to use. Default is 1000, but use more for better accuracy.")

parser.add_argument("--ext_num",
                    type=str,
                    default=999,
                    help="str: The extension number for the output files. Should take the form of 00x, 0xy, xyz.")

args = parser.parse_args()

print("Rcut = " + str(args.rcut) + "ntoys = " + str(args.ntoys) + ", extension number = " + str(args.ext_num))

# =========================== Load and prepare data ============================

data_dir = 'Data/'
vae_outputs = 'vae_outputs/'
plot_dir = 'Plots/'
model_dir = 'models/'

use_kde = False

fig_specification = 'chwzp05'
if not os.path.isdir(plot_dir): os.system('mkdir '+ plot_dir)
if not os.path.isdir(model_dir): os.system('mkdir '+ model_dir)

plt.close("all")
sm_recon_error = np.loadtxt(vae_outputs + "vh_chw_zero_recons_zp005_cHW_normalised_001.txt")
#sm_recon_error = np.loadtxt("vh_chw_zero_recons_zp005_cHW_normalised_001.txt")
eft_recon_error = np.loadtxt(vae_outputs + "vh_chw_zp005_recons_zp005_cHW_normalised_001.txt")

# =========================== Find and plot pdf ================================
extension = 'with_poisson_' + str(args.rcut) + 'Pcut_' + str(int(args.ntoys/1000)) + 'ktoys_general' + str(args.ext_num)
plot_dir = 'Plots/' + extension + '/'
array_dir = 'arrays/'
fig_specification = ''
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(array_dir, exist_ok=True)
plt.close("all")

nbins = 100
min_bin = min(np.min(sm_recon_error), np.min(eft_recon_error))
#max_bin = max(np.max(sm_recon_error), np.max(eft_recon_error))
max_bin = min(np.max(sm_recon_error), np.max(eft_recon_error))

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(sm_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.5)
ax.hist(eft_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'EFT', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel(r'R')
ax.set_title("SM vs EFT")
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
sm_reference_pdf, sm_bins = get_pdf(sm_recon_error, 0, max_bin, nbins)
eft_reference_pdf, eft_bins = get_pdf(eft_recon_error, 0, max_bin, nbins)

# Center bins
sm_bins_centered = np.zeros(len(sm_bins) - 1)
for i in range(len(sm_bins) - 1):
    sm_bins_centered[i] = (sm_bins[i] + sm_bins[i+1])/2

eft_bins_centered = np.zeros(len(eft_bins) - 1)
for i in range(len(eft_bins) - 1):
    eft_bins_centered[i] = (eft_bins[i] + eft_bins[i+1])/2
mixed_bins_centered = eft_bins_centered

# Check that the pdf we will use in the computation is the same as the one used for visualisation
fig, ax = plt.subplots(1,1, figsize = (8,8))
#plt.hist(sm_reference_pdf,sm_bins)
plt.plot(sm_bins_centered , sm_reference_pdf)
plt.plot(eft_bins_centered , eft_reference_pdf)
ax.legend()
ax.set_xlabel(r'R')
ax.set_title("SM vs EFT")
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

mixed_sample = eft_recon_error
mixed_reference_pdf = eft_reference_pdf
mixed_bins = eft_bins

if use_kde != True:
    # Only use the PDF up to max non-zero pdf bin
    sm_reference_pdf_trim = np.stack(list(sm_reference_pdf))
    mixed_reference_pdf_trim = np.stack(list(mixed_reference_pdf))
    sm_bins_centered_trim = np.stack(list(sm_bins_centered))
    mixed_bins_centered_trim = np.stack(list(mixed_bins_centered))

    trim_point = np.where(sm_reference_pdf == 0)[0][0]
    sm_reference_pdf_trim = np.delete(sm_reference_pdf_trim,np.where(sm_bins_centered > sm_bins_centered[trim_point - 1])[0])
    mixed_reference_pdf_trim = np.delete(mixed_reference_pdf_trim,np.where(mixed_bins_centered > sm_bins_centered[trim_point - 1])[0])
    sm_bins_centered_trim = np.delete(sm_bins_centered_trim,np.where(sm_bins_centered > sm_bins_centered[trim_point - 1])[0])
    mixed_bins_centered_trim = np.delete(mixed_bins_centered_trim,np.where(mixed_bins_centered > mixed_bins_centered[trim_point - 1])[0])

    sm_reference_pdf = sm_reference_pdf_trim
    mixed_reference_pdf = mixed_reference_pdf_trim
    sm_bins_centered = sm_bins_centered_trim
    mixed_bins_centered = mixed_bins_centered_trim

sm_cross_section = 23.941
eft_cross_section = 28.0
eft_to_sm_ratio = eft_cross_section/sm_cross_section
#eft_to_sm_ratio = 0.0005

#mixed_sample, mixed_reference_pdf, mixed_bins = mix_pdfs(sm_recon_error, eft_bins_centered, eft_reference_hist, eft_to_sm_ratio)
#mixed_sample = eft_recon_error
#mixed_reference_pdf = eft_reference_pdf
#mixed_bins = eft_bins

# =============================== Replace PDF with KDE ================================
if use_kde == True:
    from sklearn.neighbors import KernelDensity

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=0.001, kernel='gaussian')
    kde.fit(sm_recon_error[:, None])
    kde2 = KernelDensity(bandwidth=0.001, kernel='gaussian')
    kde2.fit(eft_recon_error[:, None])

    logkde = kde.score_samples(sm_bins_centered[:, None])
    kdefit = np.exp(logkde)
    logkde2 = kde2.score_samples(sm_bins_centered[:, None])
    kdefit2 = np.exp(logkde2)

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    #plt.hist(qcd_reference_pdf,sm_bins)
    plt.plot(sm_bins_centered , kdefit)
    plt.plot(sm_bins_centered , kdefit2)
    ax.legend()
    ax.set_xlabel('P(Top Jet)')
    ax.set_title("KDE QCD vs Top")
    ax.set_xlim()

    # Set KDE to be the PDF
    sm_reference_pdf = kdefit
    mixed_reference_pdf = kdefit2


# ====================== Find and plot mixed pdf ===============================
"""

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
#ax.hist(sm_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.5)
ax.hist(eft_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'EFT', density = True, alpha = 0.5)
ax.hist(mixed_sample, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel(r'R')
ax.set_title("SM vs EFT")
ax.set_xlim()
plt.savefig(plot_dir + "pdf_plot" + fig_specification + ".pdf")


# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
#ax.hist(sm_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.5)
ax.hist(eft_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'EFT', density = True, alpha = 0.5)
ax.hist(mixed_sample, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel(r'R')
ax.set_title("SM vs EFT")
ax.set_xlim()
#plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

# New plots
# This is just a plot for visualisation purposes and is not necessary for the code
#fig, ax = plt.subplots(1,1, figsize = (8,8))
fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
#ax.hist(eft_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'EFT', density = True, alpha = 0.7)
ax.hist(sm_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.7)
ax.hist(mixed_sample, bins = np.linspace(0, max_bin, nbins), label = 'SM + EFT', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.set_xlabel(r'$R$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("SM vs SM + EFT",fontsize=14)
ax.set_xlim()
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
#ax.hist(eft_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'EFT', density = True, alpha = 0.7)
ax.hist(sm_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.7)
ax.hist(mixed_sample, bins = np.linspace(0, max_bin, nbins), label = 'SM + EFT', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.set_xlabel(r'$R$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("SM vs SM + EFT",fontsize=14)
ax.set_xlim()
ax.set_yscale('log')
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

# Cut PDF
cut_point = 0.5
#eft_recon_error_cut = np.stack(list(eft_recon_error))
sm_recon_error_cut = np.stack(list(sm_recon_error))
mixed_sample_cut = np.stack(list(mixed_sample))
#eft_recon_error_cut = np.delete(eft_recon_error_cut,np.where(eft_recon_error_cut < cut_point)[0])
sm_recon_error_cut = np.delete(sm_recon_error_cut,np.where(sm_recon_error_cut < cut_point)[0])
mixed_sample_cut = np.delete(mixed_sample_cut,np.where(mixed_sample_cut < cut_point)[0])

fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
#ax.hist(eft_recon_error_cut, bins = np.linspace(0, max_bin, nbins), label = 'EFT', density = True, alpha = 0.7)
ax.hist(sm_recon_error_cut, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.7)
ax.hist(mixed_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'SM + EFT', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.add_artist(l1)
ax.set_xlabel(r'$R$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("SM vs SM + EFT",fontsize=14)
ax.set_xlim(0,1)
#ax.set_yscale('log')
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

fig = plt.figure(figsize = (6,6))
ax=fig.add_axes([0.13,0.11,0.8,0.8])
#ax.hist(eft_recon_error_cut, bins = np.linspace(0, max_bin, nbins), label = 'EFT', density = True, alpha = 0.7)
ax.hist(sm_recon_error_cut, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.7)
ax.hist(mixed_sample_cut, bins = np.linspace(0, max_bin, nbins), label = 'SM + EFT', density = True, alpha = 0.7)
l1=ax.legend(loc="upper center",fontsize=14)
ax.add_artist(l1)
ax.set_xlabel(r'$R$',fontsize=14)
ax.set_ylabel(r'PDF',fontsize=14)
ax.set_title("SM vs SM + EFT",fontsize=14)
ax.set_xlim(0,1)
ax.set_yscale('log')
plt.savefig(plot_dir + "pdfs" + fig_specification + ".pdf")

"""




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

# Function to sample from toy experiments and return the log likelihoods for sm
# and eft if they were to be sampled from either eft or sm
def sample_ll_from_toys(pdfA, pdfA_bins, pdfB, N_toys=10000, N_toy_events=50):
    # For each toy experiment we will find the LLR
    toy_log_likelihood_sum_listA = []
    toy_log_likelihood_sum_listB = []

    for toy in range(N_toys):
        # Get sample bin values for pdf A, using pdf A as weights

        toy_histonum = random.choices(pdfA_bins, weights=pdfA, k=N_toy_events)

        # Get the histogram of events corresponding to the randomly sampled bin/x-axis values
        # nbins must be the same as nbins of pdf
        nbins = len(pdfA) + 1
        min_P = np.min(pdfA_bins)
        #max_P = 1
        #max_P = np.max(pdfA_bins) + np.diff(pdfA_bins)[0]/2 # Not sure why I have to do this to get the plot good, but I do
        max_P = np.max(pdfA_bins)
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

def plot_llr(LLR, anomaly_type, N_toys, N_toy_events, R_threshold):

    # Plot and get histogram of -2ln(lambda) sampled from each toy experiment
    fig = plt.figure(figsize = (6,6))
    ax=fig.add_axes([0.13,0.14,0.8,0.8])
    min_llr = np.floor(np.min(LLR))
    max_llr = np.ceil(np.max(LLR))
    nbins = 100
    LLR_histo, LLR_bins, _ = ax.hist(LLR, bins=np.linspace(min_llr,max_llr,100),label='LLR', alpha = 0.7)
    # This requires centering the bins so that we can accurately fit a Gaussian
    LLR_binscenters = np.array([0.5 * (LLR_bins[i] + LLR_bins[i+1]) for i in range(len(LLR_bins)-1)])

    #print("LLRsm_binscenters",LLRsm_binscenters)
    #print("LLRsm_histo",LLRsm_histo)

    # Fit a Gaussian
    LLR_gaus, _ = fit_gaussian(LLR, LLR_histo, LLR_binscenters)

    # Plot the Gaussian
    ax.plot(LLR_binscenters, LLR_gaus, 'C1')
    ax.set_title(r'SM vs SM + EFT, $P_\mathrm{cut}(eft)$ = %s' % R_threshold, fontsize=14)
    #ax.set_title(r'SM vs SM + EFT', fontsize=14)
    ax.set_xlabel(r'LLR')
    ax.set_ylabel(r'Frequency')
    l1=ax.legend(loc=1,fontsize=14)
    l2=ax.legend([r"$N_\mathrm{sm \: events} = %s$" "\n" "$N_\mathrm{eft \: events} = %s$" % (N_toy_events[0],N_toy_events[1]-N_toy_events[0])],loc=2,prop={'size':14},handlelength=0,handletextpad=0)
    ax.add_artist(l1)
    ax.set_xlabel('LLR',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.savefig(plot_dir + "SM{}_LLR_{}toy_events".format(anomaly_type, N_toy_events) + fig_specification + ".pdf")

    return LLR_histo, LLR_bins

# ================================== Some binning stuff ======================================================

# A check that area of pdf = 1:
sm_bin_width = sm_bins[1] - sm_bins[0]
eft_bin_width = eft_bins[1] - eft_bins[0]
print("Area of unscaled sm pdf =",sm_reference_pdf.sum()*sm_bin_width)
print("Area of unscaled eft pdf =",eft_reference_pdf.sum()*eft_bin_width)

mixed_bin_width = mixed_bins[1] - mixed_bins[0]

# =================================== Plot -2nln(lambda) for an individual toy ==================================
# I have removed this to keep the code concise but if we want to add it we can adapt it from an older version


# = Run the scipt to get LLR distributions, and calcualte seperation for a number of toy experiments ======================

def run_toys_luminosity(luminosity, R_threshold, sm_cross_section, eft_cross_section, detector_efficiency):
    # Get total number of events - these will be used as means in the Poisson distributions
    mean_N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    mean_N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section) # N will be greater for mixed? Well this is the proper way of doing it - luminosity is the thing controlled, not number of events

    # Get n events for each toy as a number sampled from the poisson distribution with some mean
    # This list will be used for n events in both poisson factor and probs pdf factor
    N_toy_sm_events_list = []
    N_toy_mixed_events_list = []
    for toy in range(N_toys):
        N_toy_sm_events = np.random.poisson(mean_N_toy_sm_events)
        N_toy_mixed_events = np.random.poisson(mean_N_toy_mixed_events)
        N_toy_sm_events_list.append(N_toy_sm_events)
        N_toy_mixed_events_list.append(N_toy_mixed_events)
    #print(N_toy_sm_events_list)
    #print(N_toy_mixed_events_list)

    print("\n\n")
    print("Rcut:",R_threshold)
    print("N SM toys",mean_N_toy_sm_events)
    print("N mixed toys",mean_N_toy_mixed_events)

    # Make a cut on the pdf
    print("fraction of SM pdf before:",sm_reference_pdf.sum()*sm_bin_width)
    print("fraction of mixed pdf before cut:",mixed_reference_pdf.sum()*eft_bin_width)

    # Copy and cut on pdf and bins
    sm_reference_pdf_cut = np.stack(list(sm_reference_pdf))
    mixed_reference_pdf_cut = np.stack(list(mixed_reference_pdf))
    sm_bins_centered_cut = np.stack(list(sm_bins_centered))
    mixed_bins_centered_cut = np.stack(list(mixed_bins_centered))

    sm_reference_pdf_cut = np.delete(sm_reference_pdf_cut,np.where(sm_bins_centered < R_threshold)[0])
    mixed_reference_pdf_cut = np.delete(mixed_reference_pdf_cut,np.where(mixed_bins_centered < R_threshold)[0])
    sm_bins_centered_cut = np.delete(sm_bins_centered_cut,np.where(sm_bins_centered < R_threshold)[0])
    mixed_bins_centered_cut = np.delete(mixed_bins_centered_cut,np.where(mixed_bins_centered < R_threshold)[0])

    # Reduce number of events if cutting
    epsilon_sm = sm_reference_pdf_cut.sum()*sm_bin_width
    epsilon_mixed = mixed_reference_pdf_cut.sum()*mixed_bin_width

    mean_N_sm_after_cut = int(epsilon_sm*mean_N_toy_sm_events)
    mean_N_mixed_after_cut = int(epsilon_mixed*mean_N_toy_mixed_events)

    N_sm_after_cut_list = []
    N_mixed_after_cut_list = []
    for toy in range(N_toys):
        N_sm_after_cut = int(epsilon_sm*N_toy_sm_events_list[toy])
        N_mixed_after_cut = int(epsilon_mixed*N_toy_mixed_events_list[toy])
        N_sm_after_cut_list.append(N_sm_after_cut)
        N_mixed_after_cut_list.append(N_mixed_after_cut)

        #print("N SM before cut:", N_toy_sm_events_list[toy], "N SM after cut:", N_sm_after_cut, "fraction of SM pdf remaining after cut:",sm_reference_pdf_cut.sum()*sm_bin_width) # this will be wrong now, but it is just a print
        #print("N mixed before cut:", N_toy_mixed_events_list[toy], "N mixed after cut:", N_mixed_after_cut, "fraction of mixed pdf remaining after cut:",mixed_reference_pdf_cut.sum()*mixed_bin_width)# this will be wrong now, but it is just a print

    print("The sm bins that will be sampled from:",sm_bins_centered_cut)
    print("The mixed bins that will be sampled from:",mixed_bins_centered_cut)
    print("The sm pdfs that will be sampled from:",sm_reference_pdf_cut)
    print("The mixed pdfs that will be sampled from:",mixed_reference_pdf_cut)

    # Renorm pdfs
    sm_reference_pdf_cut = sm_reference_pdf_cut*(1.0/(sm_reference_pdf_cut.sum()*np.diff(sm_bins_centered_cut)[0]))
    mixed_reference_pdf_cut = mixed_reference_pdf_cut*(1.0/(mixed_reference_pdf_cut.sum()*np.diff(mixed_bins_centered_cut)[0]))
    print("PDF RENORMALISATION CHECK",sm_reference_pdf_cut.sum()*sm_bin_width)
    print("PDF RENORMALISATION CHECK",mixed_reference_pdf_cut.sum()*mixed_bin_width)

    # Plot PDFs after making a cut - this method is bad because matplotlib bar sucks for plotting
    plot_cut_pdfs = False
    if plot_cut_pdfs == True:
        fig, ax = plt.subplots(1,1, figsize = (8,8))
        #ax.hist(sm_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'SM', density = True, alpha = 0.5)
        #ax.hist(mixed_reference_pdf_cut, bins = np.linspace(0, max_bin, nbins), label = 'Mixed', density = True, alpha = 0.5)
        ax.bar(mixed_bins_centered_cut, mixed_reference_pdf_cut, width=np.diff(mixed_bins_centered_cut)[0], label = 'SM+EFT', alpha = 0.7)
        ax.bar(sm_bins_centered_cut, sm_reference_pdf_cut, width=np.diff(sm_bins_centered_cut)[0], label = 'SM', alpha = 0.7)
        ax.legend()
        ax.set_xlabel(r'$R$')
        ax.set_title(r"SM vs EFT $P_\mathrm{cut}(\mathrm{EFT}) = %s$" % R_threshold)
        ax.set_xlim()
        plt.savefig(plot_dir + "pdfs_with_rcut" + str(R_threshold) + 'Rcut' + fig_specification + ".pdf")

    # Sample toys - here for the general hypothesis test we will only sample toys
    # so as to find an average for the second LLR term - beyond that they serve no purpose
    # We will also keep the value for n fixed in these toys when it was not in the simple case

    # Still a little unsure on whether n should be the same in both terms or different.
    # I am going to go with it being the same (it is a measured quantity which will
    # be (on average) the mean number of signal events)

    # Here we will be only interested in sm_sample_toy_log_likelihoodsm - eft_sample_toy_log_likelihoodeft, but both with the same N corresponding to eft
    # This may be a little confusing, and I could rewrite it in a clearer way but it is doing just L(0)/L(hat(hat)) = L(N,SM)/L(N,EFT)

    # Or maybe it is the other way round eft_sample_toy_log_likelihoodeft - m_sample_toy_log_likelihoodsm

    # Or maybe it should just be the original eft_sample_toy_log_likelihoodsm - eft_sample_toy_log_likelihoodeft - it probably should then it works well

    # Actually I should really revisit this.
    cut_probs_pdf = False
    # If cutting on probs PDF
    if cut_probs_pdf == True:
        N0 = mean_N_sm_after_cut # Not sure about this but try it
        N = mean_N_mixed_after_cut
        sm_sample_toy_log_likelihoodsm, sm_sample_toy_log_likelihoodeft = sample_ll_from_toys(sm_reference_pdf_cut,sm_bins_centered_cut,mixed_reference_pdf_cut, N_toys = N_toys, N_toy_events = N0)
        eft_sample_toy_log_likelihoodeft, eft_sample_toy_log_likelihoodsm = sample_ll_from_toys(mixed_reference_pdf_cut,mixed_bins_centered_cut, sm_reference_pdf_cut, N_toys = N_toys, N_toy_events = N)
        #sm_sample_toy_log_likelihoodsm, sm_sample_toy_log_likelihoodeft = sample_ll_from_toys(sm_reference_pdf_cut,sm_bins_centered_cut,mixed_reference_pdf_cut, N_toys = N_toys, N_toy_events = N)
        #eft_sample_toy_log_likelihoodeft, eft_sample_toy_log_likelihoodsm = sample_ll_from_toys(mixed_reference_pdf_cut,mixed_bins_centered_cut, sm_reference_pdf_cut, N_toys = N_toys, N_toy_events = N)

    # If not cutting on probs PDF
    elif cut_probs_pdf != True:
        N0 = mean_N_toy_sm_events
        N = mean_N_toy_mixed_events
        sm_sample_toy_log_likelihoodsm, sm_sample_toy_log_likelihoodeft = sample_ll_from_toys(sm_reference_pdf,sm_bins_centered,mixed_reference_pdf, N_toys = N_toys, N_toy_events = N0)
        eft_sample_toy_log_likelihoodeft, eft_sample_toy_log_likelihoodsm = sample_ll_from_toys(mixed_reference_pdf,mixed_bins_centered, sm_reference_pdf, N_toys = N_toys, N_toy_events = N)
        #sm_sample_toy_log_likelihoodsm, sm_sample_toy_log_likelihoodeft = sample_ll_from_toys(sm_reference_pdf,sm_bins_centered,mixed_reference_pdf, N_toys = N_toys, N_toy_events = N)
        #eft_sample_toy_log_likelihoodeft, eft_sample_toy_log_likelihoodsm = sample_ll_from_toys(mixed_reference_pdf,mixed_bins_centered, sm_reference_pdf, N_toys = N_toys, N_toy_events = N)

    print("fraction of SM pdf remaining after cut:",sm_reference_pdf_cut.sum()*sm_bin_width)
    print("fraction of mixed pdf remaining after cut:",mixed_reference_pdf_cut.sum()*eft_bin_width)

    # Calculate Poisson factor
    poisson_events_from_probs_cut = True
    if poisson_events_from_probs_cut == True:
        N = mean_N_mixed_after_cut
        b = mean_N_sm_after_cut
        t = 10
        bhh = (N + (t*b))/(1+t) # This isn't a full implementation btw - I would need to adjust the other term too - or would I...? Actually probably not. Not for bhh anyway.
        mu = mean_N_mixed_after_cut - mean_N_sm_after_cut
    elif poisson_events_from_probs_cut != True:
        N = mean_N_toy_mixed_events
        b = mean_N_toy_sm_events
        mu = mean_N_toy_mixed_events - mean_N_toy_sm_events


    LLR_poisson0 = -2*(N*log(b/(mu + b)) + mu)
    LLR_poisson = -2*(N*log(bhh/(mu+b)) - bhh + (mu + b))
    #print("LLR sm Poisson factor:", LLR_sm_poisson)
    #print("LLR mixed Poisson factor:", LLR_mixed_poisson)

    print(b,bhh)
    print(LLR_poisson0)
    print(LLR_poisson)

    # Calculate ratio
    LLR_list = []
    LLR_second_term_list = []
    for i in range(len(sm_sample_toy_log_likelihoodsm)):
        #LLR_second_term = sm_sample_toy_log_likelihoodsm[i] - eft_sample_toy_log_likelihoodeft[i]
        #LLR_second_term = eft_sample_toy_log_likelihoodsm[i] - sm_sample_toy_log_likelihoodeft[i] # WHICH WAY - REVISIT THIS - it's also quite high
        LLR_second_term = eft_sample_toy_log_likelihoodsm[i] - eft_sample_toy_log_likelihoodeft[i] # WHICH WAY - REVISIT THIS
        LLR = LLR_poisson + LLR_second_term
        #print(LLR_poisson, LLR_second_term)

        # Test with only Poisson factor
        #LLRsm = LLR_sm_poisson_list[i]
        #LLReft = LLR_mixed_poisson_list[i]

        #print("\n",LLR_sm_poisson_list[i], sm_sample_toy_log_likelihoodsm[i]-sm_sample_toy_log_likelihoodeft[i], LLRsm)
        #print(LLR_mixed_poisson_list[i], eft_sample_toy_log_likelihoodsm[i]-eft_sample_toy_log_likelihoodeft[i], LLReft)
        LLR_list.append(LLR)
        LLR_second_term_list.append(LLR_second_term)

    avg_LLR_second_term = np.average(LLR_second_term_list)

    final_LLR = LLR_poisson + avg_LLR_second_term

    # Plot
    LLR_histo, LLR_bins= plot_llr(LLR_second_term_list, 'eft', N_toys = 10000, N_toy_events = (mean_N_toy_sm_events, mean_N_toy_mixed_events), R_threshold=R_threshold)

    print("LLR",LLR_poisson, avg_LLR_second_term, final_LLR)

    hello = "hi"
    return hello

# ========================== Z vs Neft ===========================================

# During development, we may set a cross section ratio instead of using the actual cross sections. So get the eft cross section here
#eft_cross_section = eft_to_sm_ratio*sm_cross_section

# Convert cross sections from pb to fb
#sm_cross_section = sm_cross_section*10**3 # Already done
#eft_cross_section = eft_cross_section*10**3

detector_efficiency = 1
N_toys = args.ntoys


R_threshold = args.rcut
luminosity_arr = np.linspace(0.1, 20.0, 5)
nstdevs_list = []
nstdevs_no_beta_list = []
nstdevs_exact_list = []
for luminosity in luminosity_arr:
    hi = run_toys_luminosity(luminosity, R_threshold, sm_cross_section, eft_cross_section, detector_efficiency)

    """
    nstdevs_list.append(nstdevs)
    nstdevs_no_beta_list.append(nstdevs_no_beta)
    nstdevs_exact_list.append(nstdevs_exact)


# Save the arrays so we can plot them later

nstdevs_arr = np.stack((nstdevs_list))
nstdevs_no_beta_arr = np.stack((nstdevs_no_beta_list))
nstdevs_exact_arr = np.stack((nstdevs_exact_list))
np.savetxt(array_dir + 'luminosityZvsNeft_arr' + extension + '.txt', luminosity_arr)
np.savetxt(array_dir + 'nstdevsZvsNeft_arr' + extension + '.txt', nstdevs_arr)
np.savetxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension + '.txt', nstdevs_no_beta_arr)
np.savetxt(array_dir + 'nstdevs_exactZvsNeft_arr' + extension + '.txt', nstdevs_exact_arr)

# Load them in again
luminosity_arr = np.loadtxt(array_dir + 'luminosityZvsNeft_arr' + extension + '.txt')
nstdevs_arr = np.loadtxt(array_dir + 'nstdevsZvsNeft_arr' + extension + '.txt')
nstdevs_no_beta_arr = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNtop_arr' + extension + '.txt')
nstdevs_exact_arr = np.loadtxt(array_dir + 'nstdevs_exactZvsNeft_arr' + extension + '.txt')

plt.figure()
#N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
plt.plot(luminosity_arr, nstdevs_arr,label = r'Approx')
plt.plot(luminosity_arr, nstdevs_exact_arr,label = r'Exact')
plt.legend()
plt.xlabel(r'$L$')
plt.ylabel(r'Significance $Z$')

plt.figure()
#N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
#N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
N_toy_sm_events_list = []
N_toy_eft_events_list = []
for luminosity in luminosity_arr:
    #N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    N_toy_eft_events = int(luminosity*detector_efficiency*eft_cross_section)
    #N_toy_sm_events_list.append(N_toy_sm_events)
    N_toy_eft_events_list.append(N_toy_eft_events)
plt.plot(N_toy_eft_events_list, nstdevs_arr,label = r'Approx $N_\mathrm{SM} = $, $N_\mathrm{EFT} = $')
plt.plot(N_toy_eft_events_list, nstdevs_exact_arr,label = r'Exact $N_\mathrm{SM} =s$, $N_\mathrm{EFT} = $')
plt.legend()
plt.xlabel(r'$N_{eft}$')
plt.ylabel(r'Significance $Z$')
"""
# =========================== Z vs Pcut ===========================================
"""
luminosity_arr = np.linspace(5,5,1)
min_R = 0
#max_R = max(np.max(x_test_reconerror), np.max(anomaly_x_test_reconerror))
max_R = np.max(sm_recon_error) # There normal LLR values will not be taken past this point so make the max R this, not the max of both (which is of anomaly)
max_R = 0.02
R_threshold_arr = np.linspace(min_R, max_R,4)

plt.figure()
nstdevs_list_list = []
nstdevs_exact_list_list = []
for luminosity in luminosity_arr:
    alpha_list = []
    nstdevs_list = []
    alpha_exact_list = []
    nstdevs_exact_list = []
    for R_threshold in R_threshold_arr:
        alpha, nstdevs, alpha_exact, nstdevs_exact = run_toys_luminosity(luminosity, R_threshold, sm_cross_section, eft_cross_section, detector_efficiency)
        alpha_list.append(alpha)
        nstdevs_list.append(nstdevs)
        alpha_exact_list.append(alpha_exact)
        nstdevs_exact_list.append(nstdevs_exact)
    nstdevs_list_list.append(nstdevs_list)
    nstdevs_exact_list_list.append(nstdevs_exact_list)

# Save the arrays so we can plot them later
extension = '5L' + extension
nstdevs_arr = np.stack((nstdevs_list_list))
nstdevs_exact_arr = np.stack((nstdevs_exact_list_list))
np.savetxt(array_dir + 'testluminosityZvsPcut_arr' + extension + '.txt', luminosity_arr)
np.savetxt(array_dir + 'testR_thresholdZvsPcut_arr' + extension + '.txt', R_threshold_arr)
np.savetxt(array_dir + 'testnstdevsZvsPcut_arr' + extension + '.txt', nstdevs_arr)
np.savetxt(array_dir + 'testnstdevs_exactZvsPcut_arr' + extension + '.txt', nstdevs_exact_arr)

# Load them in again
luminosity_arr = np.loadtxt(array_dir + 'testluminosityZvsPcut_arr' + extension + '.txt')
R_threshold_arr = np.loadtxt(array_dir + 'testR_thresholdZvsPcut_arr' + extension + '.txt')
nstdevs_arr = np.loadtxt(array_dir + 'testnstdevsZvsPcut_arr' + extension + '.txt')
nstdevs_exact_arr = np.loadtxt(array_dir + 'testnstdevs_exactZvsPcut_arr' + extension + '.txt')

plt.figure()
if luminosity_arr.size > 1:
    for i,luminosity in enumerate(luminosity_arr):
        N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
        N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
        plt.plot(R_threshold_arr, nstdevs_arr[i],label = r'Approx $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_sm_events, N_toy_sm_events))
        plt.plot(R_threshold_arr, nstdevs_exact_arr[i],label = r'Exact $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_sm_events, N_toy_sm_events))
        plt.legend()
        plt.xlabel(r'$P_{cut}$')
        plt.ylabel(r'Significance $Z$')
        #plt.ylim(0,10)
else:
    N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
    plt.plot(R_threshold_arr, nstdevs_arr,label = r'Approx $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_sm_events, N_toy_sm_events))
    plt.plot(R_threshold_arr, nstdevs_arr,label = r'Exact $N_\mathrm{SM} = %s$, $N_\mathrm{EFT} = %s$' % (N_toy_sm_events, N_toy_sm_events))
    plt.legend()
    plt.xlabel(r'$P_{cut}$')
    plt.ylabel(r'Significance $Z$')
    #plt.ylim(0,10)

"""
# Run for specific values only for LLR plotting purposes
#luminosity = 2.0
#R_threshold = 0
#alpha, nstdevs, alpha_exact, nstdevs_exact = run_toys_luminosity(luminosity, R_threshold, sm_cross_section, eft_cross_section, detector_efficiency)

#plt.show(block=False)
plt.show()
