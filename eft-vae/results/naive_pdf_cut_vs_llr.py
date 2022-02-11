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
plt.close("all")
sm_recon_error = np.loadtxt("vh_chw_zero_recons_zp005_cHW_normalised_001.txt")
#sm_recon_error = np.loadtxt("vh_chw_zero_recons_zp005_cHW_normalised_001.txt")
eft_recon_error = np.loadtxt("vh_chw_zp005_recons_zp005_cHW_normalised_001.txt")

detector_efficiency = 1
# =========================== Find and plot pdf ================================
plot_dir = 'testPlots/naive_selection_cut/'
fig_specification = 'newtest001'
fig_specification = 'chwzp05'
if not os.path.isdir(plot_dir): os.system('mkdir '+ plot_dir)

plt.close("all")

nbins = 100
min_bin = min(np.min(sm_recon_error), np.min(eft_recon_error))
max_bin = min(np.max(sm_recon_error), np.max(eft_recon_error))

# This is just a plot for visualisation purposes and is not necessary for the code
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.hist(sm_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'sm', density = True, alpha = 0.5)
ax.hist(eft_recon_error, bins = np.linspace(0, max_bin, nbins), label = 'eft', density = True, alpha = 0.5)
ax.legend()
ax.set_xlabel('P(eft Jet)')
ax.set_title("Binary classification for sm vs eft")
ax.set_xlim()
#plt.savefig(plot_dir + "Firstplot" + fig_specification + ".png")

# =========================== Find and plot pdf ================================
extension = '_KDE001'
plot_dir = 'Plots/' + extension + '/'
fig_specification = ''
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

# Only use the PDF up to max non-zero pdf bin
sm_reference_pdf_trim = np.stack(list(sm_reference_pdf))
mixed_reference_pdf_trim = np.stack(list(mixed_reference_pdf))
sm_bins_centered_trim = np.stack(list(sm_bins_centered))
mixed_bins_centered_trim = np.stack(list(mixed_bins_centered))
eft_reference_pdf_trim = np.stack(list(eft_reference_pdf))
eft_bins_centered_trim = np.stack(list(eft_bins_centered))

trim_point = np.where(sm_reference_pdf == 0)[0][0]
sm_reference_pdf_trim = np.delete(sm_reference_pdf_trim,np.where(sm_bins_centered > sm_bins_centered[trim_point - 1])[0])
mixed_reference_pdf_trim = np.delete(mixed_reference_pdf_trim,np.where(mixed_bins_centered > sm_bins_centered[trim_point - 1])[0])
sm_bins_centered_trim = np.delete(sm_bins_centered_trim,np.where(sm_bins_centered > sm_bins_centered[trim_point - 1])[0])
mixed_bins_centered_trim = np.delete(mixed_bins_centered_trim,np.where(mixed_bins_centered > mixed_bins_centered[trim_point - 1])[0])
eft_reference_pdf_trim = np.delete(eft_reference_pdf_trim,np.where(eft_bins_centered > eft_bins_centered[trim_point - 1])[0])
eft_bins_centered_trim = np.delete(eft_bins_centered_trim,np.where(eft_bins_centered > eft_bins_centered[trim_point - 1])[0])

sm_reference_pdf = sm_reference_pdf_trim
mixed_reference_pdf = mixed_reference_pdf_trim
sm_bins_centered = sm_bins_centered_trim
mixed_bins_centered = mixed_bins_centered_trim
eft_reference_pdf = eft_reference_pdf_trim
eft_bins_centered = eft_bins_centered_trim


# Renormalise
sm_reference_pdf = sm_reference_pdf*(1.0/(sm_reference_pdf.sum()*np.diff(sm_bins_centered)[0]))
mixed_reference_pdf = eft_reference_pdf*(1.0/(mixed_reference_pdf.sum()*np.diff(mixed_bins_centered)[0]))
eft_reference_pdf = eft_reference_pdf*(1.0/(eft_reference_pdf.sum()*np.diff(eft_bins_centered)[0]))



#sm_cross_section = 23.941
#eft_cross_section = 28.0
sm_cross_section = 0.014009*1000
eft_cross_section = 0.017125*1000
eft_to_sm_ratio = eft_cross_section/sm_cross_section
#eft_to_sm_ratio = 0.0005

#mixed_sample, mixed_reference_pdf, mixed_bins = mix_pdfs(sm_recon_error, eft_bins_centered, eft_reference_hist, eft_to_sm_ratio)
#mixed_sample = eft_recon_error
#mixed_reference_pdf = eft_reference_pdf
#mixed_bins = eft_bins


# Convert cross sections from pb to fb
#sm_cross_section = sm_cross_section*10**3
#eft_cross_section = eft_cross_section*10**3

sm_bin_width = sm_bins[1] - sm_bins[0]
eft_bin_width = eft_bins[1] - eft_bins[0]

print("fraction of sm pdf before:",sm_reference_pdf.sum()*sm_bin_width)
print("fraction of eft pdf before cut:",eft_reference_pdf.sum()*eft_bin_width)

criterion = 'criterion3'

def get_sigma(luminosity, R_threshold, criteria_type):

    sm_reference_pdf_cut = np.stack(list(sm_reference_pdf))
    eft_reference_pdf_cut = np.stack(list(eft_reference_pdf))
    mixed_reference_pdf_cut = np.stack(list(mixed_reference_pdf))

    sm_reference_pdf_cut[np.where(sm_bins_centered < R_threshold)[0]] = 0
    eft_reference_pdf_cut[np.where(sm_bins_centered < R_threshold)[0]] = 0
    mixed_reference_pdf_cut[np.where(sm_bins_centered < R_threshold)[0]] = 0

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
    print("S", N_pure_eft_after_cut, "B", N_sm_after_cut, "S+B", N_eft_after_cut, "T1", (N_pure_eft_after_cut + N_sm_after_cut), "T2", np.log(1.0 + N_pure_eft_after_cut/N_sm_after_cut))
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
R_threshold = 0
sigma_criteria1_list = []
sigma_criteria2_list = []
sigma_criteria3_list = []
for luminosity in luminosity_arr:
    sigma_criteria1 = get_sigma(luminosity, R_threshold, "1")
    sigma_criteria2 = get_sigma(luminosity, R_threshold, "2")
    sigma_criteria3 = get_sigma(luminosity, R_threshold, "3")
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

# Define smoothing filter
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# Load LLR arrays
#array_dir = 'arrays/'
array_dir = '../arrays/'
llr_Rcut = '0.0'
#extension = 'with_poisson_0.0Pcut_100ktoys_general500bins001'
extension = 'with_poisson_0Pcut_10ktoys_general002'
llr_luminosity_arr = np.loadtxt(array_dir + 'luminosityZvsNeft_arr' + extension + '.txt')
llr_nstdevs_arr = np.loadtxt(array_dir + 'nstdevsZvsNeft_arr' + extension + '.txt')

llr_nstdevs_arr[0] = 0

# A quick hack and smoothing
#llr_nstdevs_arr[5] = (llr_nstdevs_arr[6]+llr_nstdevs_arr[4])/2
llr_nstdevs_smoothed_arr = savitzky_golay(llr_nstdevs_arr, 5, 2)

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

#plt.plot(N_toy_eft_events_list, llr_nstdevs_arr,label = r'Approx')
#plt.plot(N_toy_eft_events_list, llr_nstdevs_smoothed_arr,label = r'%s $P_\mathrm{cut}$ Fisher' % llr_Rcut,color='m')

#plt.axhline(y=3, color='k', linestyle='dotted',label=r'$\alpha=1.35 \times 10^{-3}$', linewidth = 3)
#plt.axhline(y=5, color='k', linestyle='dashed',label=r'$\alpha=2.87 \times 10^{-7}$', linewidth = 3)

plt.plot(N_toy_eft_events_list, llr_nstdevs_arr,label = r'$Z = Z(\langle \Lambda_{H_1} \rangle)$',color='c', linewidth = 3)
#plt.plot(N_toy_eft_events_list, llr_nstdevs_exact_arr,label = r'Exact')


#ax.plot(N_toy_eft_events_list0, sigma_criteria1_list, label=r'%.1f $R_\mathrm{cut}$ $Z = S/\sqrt{B}$' % R_threshold,color='g',linestyle='solid')
#ax.plot(N_toy_eft_events_list0, sigma_criteria2_list, label=r'%.1f $R_\mathrm{cut}$ $Z = S/\sqrt{S + B}$' % R_threshold,color='b',linestyle='dashed')
ax.plot(N_toy_eft_events_list0, sigma_criteria3_list, label=r'$Z$ = Asimov',color='y',linestyle='dotted', linewidth = 3)

# ==================== Add as well the supervised lines =================================
# Load LLR arrays
array_dir = '../arrays/supervised/'
llr_Pcut = '0.0'
#extension = 'with_poisson_0.0Pcut_100ktoysfloat_new_xsec001'
extension = 'with_poisson_0Pcut_100ktoys999'
llr_luminosity_arr_sup = np.loadtxt(array_dir + 'luminosityZvsNeft_arr' + extension + '.txt')
llr_nstdevs_arr_sup = np.loadtxt(array_dir + 'nstdevsZvsNeft_arr' + extension + '.txt')
llr_nstdevs_no_beta_arr_sup = np.loadtxt(array_dir + 'nstdevs_no_betaZvsNeft_arr' + extension + '.txt')
llr_nstdevs_exact_arr_sup = np.loadtxt(array_dir + 'nstdevs_exactZvsNeft_arr' + extension + '.txt')

N_toy_sm_events_list_sup = []
N_toy_eft_events_list_sup = []
for luminosity in llr_luminosity_arr_sup:
    #N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    #N_toy_eft_events = luminosity*detector_efficiency*(eft_cross_section - sm_cross_section)
    N_toy_eft_events_sup = luminosity*detector_efficiency*(eft_cross_section)
    #N_toy_sm_events_list.append(N_toy_sm_events)
    N_toy_eft_events_list_sup.append(N_toy_eft_events_sup)

plt.plot(N_toy_eft_events_list_sup, llr_nstdevs_arr_sup,label = r'$\alpha=$ Supervised symmetric',color='m', linestyle = 'dashed', linewidth = 3)
plt.plot(N_toy_eft_events_list_sup, llr_nstdevs_no_beta_arr_sup,label = r'$\alpha=$ Supervised asymmetric',color='m', linestyle = 'solid', linewidth = 3)


# ===================================================================================

plt.legend()
plt.xlabel(r'$N_\mathrm{EFT}$',fontsize=18)
plt.ylabel(r'Significance',fontsize=18)
#ax.set_title(r'$R_\mathrm{cut}$ = %.1f' % R_threshold, fontsize=18)




"""
# Z vs Pcut
luminosity = 5.0
N_eft_before_cut = eft_cross_section*luminosity
N_sm_before_cut = sm_cross_section*luminosity

R_threshold_list = np.linspace(0,0.02,1000)
sigma_criteria1_list = []
sigma_criteria2_list = []
sigma_criteria3_list = []

for R_threshold in R_threshold_list:
    sigma_criteria1 = get_sigma(luminosity, R_threshold, "1")
    sigma_criteria2 = get_sigma(luminosity, R_threshold, "2")
    sigma_criteria3 = get_sigma(luminosity, R_threshold, "3")
    sigma_criteria1_list.append(sigma_criteria1)
    sigma_criteria2_list.append(sigma_criteria2)
    sigma_criteria3_list.append(sigma_criteria3)


# Load LLR significances
extension = '5L001'
llr_luminosity_arr = np.loadtxt(array_dir + 'testluminosityZvsPcut_arr' + extension + '.txt')
llr_R_threshold_arr = np.loadtxt(array_dir + 'testR_thresholdZvsPcut_arr' + extension + '.txt')
llr_nstdevs_arr = np.loadtxt(array_dir + 'testnstdevsZvsPcut_arr' + extension + '.txt')
llr_nstdevs_exact_arr = np.loadtxt(array_dir + 'testnstdevs_exactZvsPcut_arr' + extension + '.txt')



fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(R_threshold_list, sigma_criteria1_list, label=r'$S/\sqrt{B}$, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_eft_before_cut, N_sm_before_cut),alpha=0.5)
ax.plot(R_threshold_list, sigma_criteria2_list, label=r'$S/\sqrt{S + B}$, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_eft_before_cut, N_sm_before_cut),alpha=0.5)
ax.plot(R_threshold_list, sigma_criteria3_list, label=r'Asimov, L= %.2f fb$^{-1}$ , $N_{EFT}$ = %i , $N_{SM}$ = %i (before cut)' % (luminosity,N_eft_before_cut, N_sm_before_cut),alpha=0.5)

if llr_luminosity_arr.size > 1:
    for i,luminosity in enumerate(llr_luminosity_arr):
        N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
        N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
        plt.plot(llr_R_threshold_arr, llr_nstdevs_arr[i],label = r'Approx $N_\mathrm{sm} = %s$, $N_\mathrm{eft} = %s$' % (N_toy_sm_events, N_toy_mixed_events))
        #plt.plot(llr_R_threshold_arr, llr_nstdevs_exact_arr[i],label = r'Exact $N_\mathrm{sm} = %s$, $N_\mathrm{eft} = %s$' % (N_toy_sm_events, N_toy_sm_events))
        plt.legend()
        plt.xlabel(r'$R_{cut}$')
        plt.ylabel(r'Significance $Z$')
        #plt.ylim(0,10)
else:
    N_toy_sm_events = int(luminosity*detector_efficiency*sm_cross_section)
    N_toy_mixed_events = int(luminosity*detector_efficiency*eft_cross_section)
    plt.plot(llr_R_threshold_arr, llr_nstdevs_arr,label = r'Approx $N_\mathrm{sm} = %s$, $N_\mathrm{eft} = %s$' % (N_toy_sm_events, N_toy_mixed_events))
    #plt.plot(llr_R_threshold_arr, llr_nstdevs_exact_arr,label = r'$N_\mathrm{sm} = {}, $N_\mathrm{eft} = {}$'.format(N_toy_sm_events, N_toy_sm_events))
    plt.legend()
    plt.xlabel(r'$R_{cut}$')
    plt.ylabel(r'Significance $Z$')
    #plt.ylim(0,10)

#ax.plot(llr_R_threshold_arr, llr_nstdevs_arr, label=r'LLR')
ax.legend()
ax.set_xlabel(r'$R_\mathrm{cut}$')
ax.set_ylabel(r'$Z$')
"""








#plt.ion()
plt.show()
