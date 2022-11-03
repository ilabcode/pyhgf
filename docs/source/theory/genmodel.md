# The generative model of the HGF: Volatility vs. value coupling

In the generative model of the HGF, (hidden) states of the world perform Gaussian random walks in time and can produce outcomes which are perceived by an observer as inputs. States can influence each other via volatility coupling or via value coupling.

In the classical 3-level binary HGF as presented in Mathys (2011), the two states of interest, $x_2$ and $x_3$, are coupled to each other via volatility coupling, which means that for state $x_2$, the mean of the Gaussian random walk on trial $k$ is given by its previous value $x_2^{(k-1)}$, while the step size (or variance) depends on the current value of the higher level state, $x_3^{(k)}$:

$$
    x_2^{(k)} \sim \mathcal{N}(x_2^{(k)} | x_2^{(k-1)}, \, f(x_3^{(k)})),
$$

where the exact dependency is of the form

$$
    f(x_3^{(k)}) = \exp(\kappa_2 x_3^{(k)} + \omega_2).
$$

However, a higher-level state can also have influence on a lower-level state by influencing its mean. In that case, the mean of the Gaussian random walk at one level is a function not only of its own previous value, but also the current value of the higher-level state (with step size either constant or a function of another state):

$$
    x_2^{(k)} \sim \mathcal{N}(x_2^{(k)} | x_2^{(k-1)} + \alpha_{4,2} x_4^{(k)}, \, \exp(\omega_2)),
$$

which means constant step size, or

$$
    x_2^{(k)} \sim \mathcal{N}(x_2^{(k)} | x_2^{(k-1)} + \alpha_{4,2} x_4^{(k)}, \, \exp(\kappa_2 x_3^{(k)} + \omega_2)).
$$

In other words, any given state in the world can be modelled as having a volatility parent state, a value parent state, or both, or none (in which case it evolves as a Gaussian random walk around its previous value with fixed step size). Consequently, when inferring on the evolution of these states, the exact belief update equations (which include the computation of new predictions, posterior values, and prediction errors, and represent an approximate inversion of this generative model, see Mathys (2011) depend on the nature of the coupling of a given state with its parent and children states. In particular, the nodes that implement the belief updates will communicate with their value parents via value prediction errors, or **VAPE**s, and via volatility prediction errors, or **VOPE**s, with their volatility parents.

![Figure1](./images/genmod.svg)
*An example of a generative model of sensory inputs with six hidden states. Volatility coupling is depicted with dashed lines, value coupling with straight lines.*

In [Figure 1](#Figure1) we have drawn an example setup with six different environmental states and one outcome. Here, we have denoted states that function as value parents for other states as $x_i$, and states that function as volatility parents as $\check{x}_i$. Volatility coupling is depicted by curvy arrows, value coupling by straight arrows, and observable outcomes are linked to their hidden states via double arrows.

For the example illustrated in [Figure 1](#Figure1) the following equations describe the generative model:

$$
\begin{align}
u^{(k)} &\sim \mathcal{N}(u^{(k)} | x_1^{(k)}, \, \sigma_u) \\
x_1^{(k)}           &\sim \mathcal{N}(x_1^{(k)} | x_1^{(k-1)} + \alpha_{2,1} x_2^{(k)}, \, \exp(\kappa_1 \check{x}_1^{(k)} + \omega_1)) \\
\check{x}_1^{(k)}   &\sim \mathcal{N}(\check{x}_1^{(k)} | \check{x}_1^{(k-1)} + \alpha_{3,\check{1}} x_3^{(k)}, \, \exp(\omega_{\check{1}})) \\
x_2^{(k)}           &\sim \mathcal{N}(x_2^{(k)} | x_2^{(k-1)}, \, \exp(\kappa_2 \check{x}_2^{(k)} + \omega_2)) \\
\check{x}_2^{(k)}   &\sim \mathcal{N}(\check{x}_2^{(k)} | \check{x}_2^{(k-1)}, \, \exp(\omega_{\check{2}})) \\
x_3^{(k)}           &\sim \mathcal{N}(x_3^{(k)} | x_3^{(k-1)}, \, \exp(\kappa_3 \check{x}_3^{(k)} + \omega_3)) \\
\check{x}_3^{(k)}   &\sim \mathcal{N}(\check{x}_3^{(k)} | \check{x}_3^{(k-1)}, \, \exp(\omega_{\check{3}})) \\
\end{align}
$$

Note that in this example, all states that are value parents of other states (or outcomes) have their own volatility parent, while states that are volatility parents to other nodes either have a value parent (as state $\check{x}_1$), or no parents (as states $\check{x}_2$ and $\check{x}_3$). This is deliberately so, and we will see these two motifs - every state of a hierarchy has its own volatility estimation, and volatility states only have value parents - reappear in the following chapters.
