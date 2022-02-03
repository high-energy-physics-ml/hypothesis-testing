"""
    Script to produce visual aids to explaining how significance relates to area under null LLR alpha
"""
import sys, os
import numpy as np
import scipy.optimize
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns; sns.set(style="white", color_codes=True)

plt.close("all")

# Make some Gaussian distributions
mu0 = -4 #-4
std0 = 3 #3
mu1 = 4 #4
std1 = 3 #3

gaus0 = np.random.normal(mu0, std0, 1000000)
gaus1 = np.random.normal(mu1, std1, 1000000)

fig, ax = plt.subplots(1,1, figsize = (8,8))
#gaus_bins = np.linspace(-60,60,100)
gaus_bins = np.linspace(-15,15,100)
gaus0_hist, gaus0_bins, _ = ax.hist(gaus0, bins=gaus_bins, alpha=0.5)
gaus1_hist, gaus1_bins, _ = ax.hist(gaus1, bins=gaus_bins, alpha=0.5)

gaus0_binscenters = np.array([0.5 * (gaus0_bins[i] + gaus0_bins[i+1]) for i in range(len(gaus0_bins)-1)])
gaus1_binscenters = np.array([0.5 * (gaus1_bins[i] + gaus1_bins[i+1]) for i in range(len(gaus1_bins)-1)])

def fit_gaussian(xdata, ydata, xbins):
    # Find parameters of Gaussian; amplitude, mean, stdev
    amp = np.max(ydata)
    mu = np.mean(xdata)
    sigma = np.std(xdata)
    print(amp, mu, sigma)

    # Define the form of the Gaussian
    def gaussian(x, amplitude, mean, stddev):
        #return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
        return amplitude * np.exp(-(1/2)*((x - mean)/stddev)**2)
        #return 1/(stddev*np.sqrt(2*np.pi)) * np.exp(-((x - mean) / 4 / stddev)**2)

    # Fit parameters for the Gaussian defined above from the data. p0 are initial guesses
    popt, _ = scipy.optimize.curve_fit(gaussian, xbins, ydata, p0 = [amp, mu, sigma])

    # Now get the Gaussian curve with those fitted parameters
    fitted_gaussian = gaussian(xbins, *popt)
    return fitted_gaussian, popt

def gaussian(x, amplitude, mean, stddev):
    #return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
    return amplitude * np.exp(-(1/2)*((x - mean)/stddev)**2)


# Get the fit
gaus0_fit, gaus0_params = fit_gaussian(gaus0, gaus0_hist, gaus0_binscenters)
gaus1_fit, gaus1_params = fit_gaussian(gaus1, gaus1_hist, gaus1_binscenters)

# Better to do this way
gaus0_fit = gaussian(gaus0_binscenters, *gaus0_params)
gaus1_fit = gaussian(gaus1_binscenters, *gaus1_params)

ax.plot(gaus0_binscenters, gaus0_fit)
ax.plot(gaus1_binscenters, gaus1_fit)

# So that I can do (after finding alpha):

def get_alpha(gaus0_list, gaus1_list, gaus0_hist, gaus0_bins, gaus1_hist, gaus1_bins):
    # Recenter bins
    gaus0_binscenters = np.array([0.5 * (gaus0_bins[i] + gaus0_bins[i+1]) for i in range(len(gaus0_bins)-1)])
    gaus1_binscenters = np.array([0.5 * (gaus1_bins[i] + gaus1_bins[i+1]) for i in range(len(gaus1_bins)-1)])

    # Get parameters for Gaussian fit
    _, test_gaus_params = fit_gaussian(gaus0_list, gaus0_hist, gaus0_binscenters)
    _, anomaly_gaus_params = fit_gaussian(gaus1_list, gaus1_hist, gaus1_binscenters)

    test_amplitude = test_gaus_params[0]
    test_mean = test_gaus_params[1]
    test_stdev = abs(test_gaus_params[2]) # abs since for some weird numerical reason stdev can be -ve. (it does not matter since it is squared but it makes me feel uncomfortable unless it is +ve)

    anomaly_amplitude = anomaly_gaus_params[0]
    anomaly_mean = anomaly_gaus_params[1]
    anomaly_stdev = abs(anomaly_gaus_params[2])

    # Integrate over Gaussian distribution with different lambda_cut as limits
    # to get alpha for which alpha = beta
    def integrand(x, amplitude, mean, stdev):
        #return amplitude * np.exp(-((x - mean) / 4 / stdev)**2)
        return amplitude * np.exp(-(1/2)*((x - mean)/stdev)**2)


    #lam_cut_potential_values = np.linspace(-200, -80, 1000)

    # min and max lamda values (would be infinity but scipy.integrate.quad breaks when using scipy.inf)
    print("====================== VALUES ================================")
    print(test_amplitude, test_mean, test_stdev, anomaly_amplitude, anomaly_mean, anomaly_stdev)
    min_lam = anomaly_mean - 50*np.average((test_stdev, anomaly_stdev))
    max_lam = test_mean + 50*np.average((test_stdev, anomaly_stdev))
    print(min_lam, max_lam)

    lam_cut_potential_values = np.linspace(anomaly_mean - 20*np.average((test_stdev, anomaly_stdev)), test_mean + 20*np.average((test_stdev, anomaly_stdev)), 10000)

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

# Calculate alpha from any point on x-axis (to infinity)
def calc_alpha(x_min, gaus0_list, gaus1_list, gaus0_hist, gaus0_bins, gaus1_hist, gaus1_bins):
    # Get parameters for Gaussian fit
    _, test_gaus_params = fit_gaussian(gaus0_list, gaus0_hist, gaus0_binscenters)
    _, anomaly_gaus_params = fit_gaussian(gaus1_list, gaus1_hist, gaus1_binscenters)

    test_amplitude = test_gaus_params[0]
    test_mean = test_gaus_params[1]
    test_stdev = abs(test_gaus_params[2]) # abs since for some weird numerical reason stdev can be -ve. (it does not matter since it is squared but it makes me feel uncomfortable unless it is +ve)

    anomaly_amplitude = anomaly_gaus_params[0]
    anomaly_mean = anomaly_gaus_params[1]
    anomaly_stdev = abs(anomaly_gaus_params[2])

    def integrand(x, amplitude, mean, stdev):
        #return amplitude * np.exp(-((x - mean) / 4 / stdev)**2)
        return amplitude * np.exp(-(1/2)*((x - mean)/stdev)**2)

    min_lam = anomaly_mean - 50*np.average((test_stdev, anomaly_stdev))
    max_lam = test_mean + 50*np.average((test_stdev, anomaly_stdev))

    alpha_normalisation ,alpha_normalisation_error = integrate.quad(integrand,min_lam,max_lam,args=(test_amplitude, test_mean, test_stdev))
    alpha_integral1 ,alpha_integral1_error = integrate.quad(integrand,min_lam,x_min,args=(test_amplitude, test_mean, test_stdev))
    alpha = alpha_integral1/alpha_normalisation

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


alpha = 1 - get_alpha(gaus0, gaus1, gaus0_hist, gaus0_bins, gaus1_hist, gaus1_bins)

# Find a region alpha under the gaus0 curve
x_alpha_min = alpha
x_alpha_max = 15 # The highest value I am plotting - theoretically infinity
x_alpha = np.linspace(x_alpha_min, x_alpha_max, 1000)
alpha_from_fit = gaussian(x_alpha, *gaus0_params)

nstdevs = get_nstdevs(alpha)

# Find a region beta under gaus1 curve
x_beta_min = -15 # The lowest value I am plotting - theoretically -infinity
x_beta_max = alpha
x_beta = np.linspace(x_beta_min, x_beta_max, 1000)
beta_from_fit = gaussian(x_beta, *gaus1_params)

fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(gaus0_binscenters, gaus0_fit)
ax.plot(gaus1_binscenters, gaus1_fit)

# Plot line for alpha min
#ax.vlines(x_alpha_min, 0, alpha_from_fit[0], color='r', linestyles='solid')
ax.fill_between(x_alpha, alpha_from_fit, color='red', label=r"$Z = {}$" .format(round(nstdevs, 5)))
#ax.vlines(x_beta_min, 0, beta_from_fit[0], color='g', linestyles='solid')
ax.fill_between(x_beta, beta_from_fit, color='green')
ax.set_xlabel(r'$x$', fontsize = 18)
ax.set_ylabel(r'Frequency', fontsize = 18)
ax.legend(loc="best")

print("Alpha = ",alpha)



# Now the same but with alpha the area under gau0 curve from mean of gaus1 to infinity
# and having no beta

# Find a region alpha under the gaus0 curve
x_alpha_min = np.average(gaus1)
x_alpha_max = 15 # The highest value I am plotting - theoretically infinity
x_alpha = np.linspace(x_alpha_min, x_alpha_max, 1000)
alpha_from_fit = gaussian(x_alpha, *gaus0_params)

alpha = 1 - calc_alpha(x_alpha_min, gaus0, gaus1, gaus0_hist, gaus0_bins, gaus1_hist, gaus1_bins)
nstdevs = get_nstdevs(alpha)

alpha_to_gaus1_line_from_fit = gaussian(x_alpha, *gaus1_params)

fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(gaus0_binscenters, gaus0_fit)
ax.plot(gaus1_binscenters, gaus1_fit)
ax.vlines(x_alpha_min, 0, alpha_to_gaus1_line_from_fit[0], color='r', linestyles='solid', linewidth = 2)
ax.fill_between(x_alpha, alpha_from_fit, color='red', label=r"$Z = {}$" .format(round(nstdevs, 5)))
ax.set_xlabel(r'$x$', fontsize = 18)
ax.set_ylabel(r'Frequency', fontsize = 18)
ax.legend(loc="best")


# Finally plot the naive Z as a number of standard deviations away from the null
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(gaus0_binscenters, gaus0_fit)
ax.plot(gaus1_binscenters, gaus1_fit)

n_stdevs = (mu1 - mu0)/std0

# Plot lines for sigma
for n in range(0, int(np.ceil(abs((mu1 - mu0)/std0))) + 3):
    #print(n)
    x_at_std = mu0 + n*std0
    std_line = gaussian(x_at_std, *gaus0_params)
    ax.vlines(x_at_std, 0, std_line, color='r', linestyles='solid', linewidth = 2)

mu1_line = gaussian(mu1, *gaus1_params)
ax.vlines(mu1, 0, mu1_line, color='g', linestyles='solid', label=r"$Z = {}$" .format(round(nstdevs, 5)), linewidth = 2)
ax.legend(loc="best")
ax.set_xlabel(r'$x$', fontsize = 18)
ax.set_ylabel(r'Frequency', fontsize = 18)


plt.show()
