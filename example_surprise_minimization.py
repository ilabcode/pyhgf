# example_surprise_minimization.py
"""Example of surprise minimization using a model from the hgf module

This doesn't work yet as it should - probably the fault of the
optimization algorithms in scipy.optimize"""

import numpy as np
import hgf
from scipy.optimize import minimize

# Set up standard 3-level HGF for binary inputs
binstdhgf = hgf.StandardBinaryHGF(initial_mu2=0.0,
                                  initial_pi2=1.0,
                                  omega2=-2.5,
                                  kappa2=1.0,
                                  initial_mu3=1.0,
                                  initial_pi3=1.0,
                                  omega3=-8.0)

# Read binary input from Iglesias et al. (2013)
binary = np.loadtxt('binary_input.dat')

# Feed input
binstdhgf.input(binary)

# Set priors
binstdhgf.x2.omega.trans_prior_mean = -3
binstdhgf.x2.omega.trans_prior_precision = 4**-2
binstdhgf.x3.omega.trans_prior_mean = -6
binstdhgf.x3.omega.trans_prior_precision = 4**-2

# Get the objective function
binstdf = binstdhgf.neg_log_joint_function()

# Minimize the negative log-joint
binstdx0 = [param.value for param in binstdhgf.var_params]
binstdmin = minimize(binstdf, binstdx0)

# Set up standard 2-level HGF for continuous inputs
stdhgf = hgf.StandardHGF(initial_mu1=1.04,
                         initial_pi1=1e4,
                         omega1=-13.0,
                         kappa1=1,
                         initial_mu2=1,
                         initial_pi2=1e1,
                         omega2=-2,
                         omega_input=np.log(1e-4))

# Read USD-CHF data
usdchf = np.loadtxt('usdchf.dat')

# Feed input
stdhgf.input(usdchf)

# Set priors
stdhgf.x1.initial_mu.trans_prior_mean = 1.0375
stdhgf.x1.initial_mu.trans_prior_precision = 4.0625e5
stdhgf.x1.initial_pi.trans_prior_mean = -10.1111
stdhgf.x1.initial_pi.trans_prior_precision = 1
stdhgf.x1.omega.trans_prior_mean = -12.1111
stdhgf.x1.omega.trans_prior_precision = 4**-2
stdhgf.x2.initial_pi.trans_prior_mean = -2.3026
stdhgf.x2.initial_pi.trans_prior_precision = 1
stdhgf.x2.omega.trans_prior_mean = -4
stdhgf.x2.omega.trans_prior_precision = 4**-2
stdhgf.xU.omega.trans_prior_mean = -10.1111
stdhgf.xU.omega.trans_prior_precision = 2**-2

# Get the objective function
stdobjf = binstdhgf.neg_log_joint_function()

# Minimize the negative log-joint
stdx0 = [param.value for param in stdhgf.var_params]
stdmin = minimize(stdobjf, stdx0)
