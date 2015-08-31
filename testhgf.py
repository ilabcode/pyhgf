import numpy as np
import hgf


x2 = hgf.StateNode(prior_mu=1,
                   prior_pi=np.inf,
                   omega=-2.0)
x1 = hgf.StateNode(prior_mu=1.04,
                   prior_pi=np.inf,
                   omega=-12.0)
xU = hgf.InputNode(omega=0.0)

x1.add_volatility_parent(parent=x2, kappa=1)
xU.set_value_parent(parent=x1)

xU.input(0.5)

stdhgf = hgf.StandardHGF(prior_mu1=1.04,
                         prior_pi1=np.inf,
                         omega1=-12.0,
                         kappa1=1,
                         prior_mu2=1,
                         prior_pi2=np.inf,
                         omega2=-2,
                         omega_input=0.0)

stdhgf.input(0.5)
