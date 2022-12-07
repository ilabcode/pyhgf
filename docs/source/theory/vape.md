# Computations for VAPE coupling

The exact computations of the **UPDATE** depend on the nature of the coupling with the child node(s), while both the **PE step** and the **PREDICTION step** depend on the coupling with the parent node(s).

## Update Step

If Node~$i$ is the value parent of Node $i-1$, then the following update equations apply to Node~$i$:

$$
\begin{align}
\pi_i^{(k)} &= \hat{\pi}_i^{(k)} + \alpha_{i-1,i}^2 \hat{\pi}_{i-1}^{(k)}\\
\mu_i^{(k)} &= \hat{\mu}_i^{(k)} + \frac{\alpha_{i-1,i}^2 \hat{\pi}_{i-1}^{(k)}} {\alpha_{i-1,i}^2 \hat{\pi}_{i-1}^{(k)} + \hat{\pi}_{i}^{(k)}} \delta_{i-1}^{(k)}
\end{align}
$$

We note here that we can let the update of the precision happen first, and therefore use it for the update of the mean:

$$
\begin{align}
\pi_i^{(k)} &= \hat{\pi}_i^{(k)} + \alpha_{i-1,i}^2 \hat{\pi}_{i-1}^{(k)}\\
\mu_i^{(k)} &= \hat{\mu}_i^{(k)} + \frac{\alpha_{i-1,i}^2 \hat{\pi}_{i-1}^{(k)}} {\pi_i^{(k)}} \delta_{i-1}^{(k)}
\end{align}
$$

In sum, at the time of the update, Node~$i$ needs to have access to the following quantities:

* Its own predictions: $\hat{\mu}_i^{(k)}$, $\hat{\pi}_i^{(k)}$  
* Coupling strength: $\alpha_{i-1,i}$  
* From level below: $\delta_{i-1}^{(k)}$, $\hat{\pi}_{i-1}^{(k)}$  

All of these are available at the time of the update. Node~$i$ therefore only needs to receive the PE and the predicted precision from the level below to perform its update.

## Prediction Error Step

We will assume in the following, that Node~$i$ is the value child of Node $i+1$. Then the following quantities have to be sent up to Node $i+1$ (cf. necessary information from level below in a value parent):

* Predicted precision: $\hat{\pi}_{i}^{(k)}$
* Prediction error: $\delta_{i}^{(k)}$

Node~$i$ has already performed the **PREDICTION step** on the previous trial, so it has already computed the predicted precision of the current trial,~$\hat{\pi}_{i}^{(k)}$. Hence, in the **PE step**, it needs to perform only the following calculation:
$$
\begin{equation}
\delta_i^{(k)} = \mu_i^{(k)} - \hat{\mu}_i^{(k)}
\end{equation}
$$

## Prediction Step

Still assuming that Node~$i$ is the value child of Node $i+1$, the **PREDICTION step** consists of the following computations:

$$
\begin{align}
\hat{\mu}_i^{(k+1)} &= \mu_i^{(k)} + \alpha_{i,i+1} \mu_{i+1}^{(k)}\\
\hat{\pi}_i^{(k+1)} &= \frac{1}{\frac{1}{\pi_i^{(k)}} + \nu_i^{(k+1)} }
\end{align}
$$

with

$$
\begin{equation}
\nu_i^{(k+1)} = \exp(\omega_i).
\end{equation}
$$

Note that if Node~$i$ additionally has a **VOPE** parent node, the estimated volatility $\nu_i^{(k+1)}$ that enters the precision update would also depend on the posterior mean of that volatility parent (cf. **PREDICTION step** for **VOPE** coupling).

In general, the prediction of the mean will depend only on whether Node~$i$ has a value parent or not, whereas the prediction of the precision only depends on whether Node~$i$ has a volatility parent or not.

Thus, the **PREDICTION step** only depends on knowing the node's own posteriors and receiving the value parent's posterior in time before the new input arrives.
