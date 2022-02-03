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

# Define Poisson dist
#gaussian = np.random.normal(30000, 2*np.sqrt(2*np.log(2)), 100000)
gaussian = np.random.normal(30000, np.sqrt(30000), 100000)
poisson = np.random.poisson(30000, 100000)

import matplotlib.pyplot as plt
plt.close("all")
plt.figure()
plt.hist(gaussian, bins=200, alpha=0.5)
#plt.figure()
plt.hist(poisson, bins=200, alpha=0.5)

min_bin = np.min(poisson)
max_bin = np.max(poisson)
nbins = int(max_bin - min_bin)
plt.figure()
plt.hist(poisson, bins = np.linspace(min_bin, max_bin, nbins), alpha=0.5,density=True)


pdf,bins = np.histogram(poisson, bins=100, density = True)

# Test finding Poisson
n0 = 30862
n1 = 31099
l0 = 30862
l1 = 31099

#n0 = 4800
#l0 = 5000

# Find log(lambda^n * exp(-lambda) / n!) = n*log(lambda) - lambda - log(n!)

log_n0_component_list = []
for i in range(1,n0 + 1):
    log_n0_component = log(i) # Was making a mistake where this I had set to n0 instead of i! This would mean that the log ratio would be unchanged since this factor did not matter anyway but the purepdf would change drastically! :0
    log_n0_component_list.append(log_n0_component)
log_n0factorial = np.sum(log_n0_component_list)

# log factorials cancel but I include them here for completeness
p0_0 = n0*log(l0) - l0 - log_n0factorial
p0_1 = n0*log(l1) - l1 - log_n0factorial

p_0 = -2*(p0_0 - p0_1)

log_n1_component_list = []
for i in range(1,n1 + 1):
    log_n1_component = log(i) # Was making a mistake where this I had set to n1 instead of i! This would mean that the log ratio would be unchanged since this factor did not matter anyway but the purepdf would change drastically! :0
    log_n1_component_list.append(log_n1_component)
log_n1factorial = np.sum(log_n1_component_list)

# log factorials cancel but I include them here for completeness
p1_0 = n1*log(l0) - l0 - log_n1factorial
p1_1 = n1*log(l1) - l1 - log_n1factorial

p_1 = -2*(p1_0 - p1_1)

# Calculation with common factors removed
p0_one_step = -2*(n0*log(l0/l1) + (l1 - l0))
p1_one_step = -2*(n1*log(l0/l1) + (l1 - l0))


# Calculate the probability of getting 4800 or lower given a Poisson with mean 5000
# by calculating first the log poisson prob for each n <= 4800, summing them and taking exp



# Another way of finding the probability of being 4800 or lower given a Poisson with mean 5000
from scipy.stats import poisson
rv = poisson(5000)
sum = 0
for num in range(0,4800):
    sum += rv.pmf(num)
print(sum)

# Prob of being less than 30000 - should be half
rv = poisson(30000)
sum = 0
for num in range(0,30000):
    sum += rv.pmf(num)
print(sum)

# For mean = 30000, we should get
rv.pmf(30000)

# Find log ratios this way
rv0 = poisson(l0)
rv1 = poisson(l1)

# A slight mixing of subscript notation but it checks out
unlogged_p0_0_new = rv0.pmf(n0)
unlogged_p0_1_new = rv1.pmf(n0)

unlogged_p1_0_new = rv0.pmf(n1)
unlogged_p1_1_new = rv1.pmf(n1)

unlogged_p0_new = unlogged_p0_0_new/unlogged_p0_1_new
unlogged_p1_new = unlogged_p1_0_new/unlogged_p1_1_new

p0_new = -2*log(unlogged_p0_new)
p1_new = -2*log(unlogged_p1_new)

print("Old log p0 | Old log p1 | New log p0 | New log p1")
print(p_0,p_1,p0_new,p1_new)




plt.show()




plt.show()
