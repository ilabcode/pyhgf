---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(hgf_exercises)=
# Hierarchical Gaussian Filter modelling exercises

```{code-cell} ipython3
:tags: [hide-cell]

%%capture
import sys
if 'google.colab' in sys.modules:
    ! pip install pyhgf
```

```{code-cell} ipython3
from pyhgf import load_data
from pyhgf.model import HGF
import pandas as pd
import jax.numpy as jnp
import arviz as az

# load example dataset
timeseries = load_data("continuous")
```

In this notebook you are going to learn the core principles on whic the HGF is built, you will build agents that can filter new observations and update their beliefs about hidden states of the world and try to optimize them so they are getting less and less surprised about what is happening. Finally, you will create and agent that tries to optimize its behavior when facing (almost) unpredictable events.

+++

## Theory: Modeling belief updating under uncertainty
### Gaussian random walks

The generative model on which the HGF is built is a generalisation of the [Gaussian Random Walk](https://en.wikipedia.org/wiki/Random_walk#Gaussian_random_walk) (GRW). A GRW generate a new observation $x_1^{(k)}$ at each time step $k$ from a normal distribution and using the previous observation $x_1^{(k-1)}$ such as:

$$
x_1^{(k)} \sim \mathcal{N}(x_1^{(k-1)}, \sigma^2)
$$

where $\sigma^2$ is the variance of the distribution. 

```{admonition} Exercise 1
Using the equation above, write a Python code that implements a Gaussian random walk using the following parameters: $\sigma^2 = 1$ and $x_1^{(0)} = 0$.
```

+++

### Volatility coupling

+++

The HGF generalize this process by making the parameters of a GRW depends on another GRW at a higher level. This kind of dependeny is termed "coupling" and can target the volatiliy ($\sigma^2$) of the value ($\mu$), or both. 

If we take as example the two-level continuous HGF {cite:p}`2014:mathys`, the model is constituded of two states of interest, $x_1$ and $x_2$. The node $x_1$ is performing a GRW, but it is also paired with $x_2$ via *volatility coupling*. This means that for state $x_1$, the mean of the Gaussian random walk on time point $k$ is given by its previous value $x_1^{(k-1)}$, while the step size (or variance) depends on the current value of the higher level state, $x_2^{(k)}$.

$$
x_1^{(k)} \sim \mathcal{N}(x_1^{(k)} | x_1^{(k-1)}, \, f(x_2^{(k)}))
$$

where the exact dependency is of the form

$$
    f(x_2^{(k)}) = \exp(x_2^{(k)} + \omega_1)
$$

At the higher level of the hierarchy (here the second level), the nodes are not inheriting anything from their parents anymore, and only rely on their own variance:

$$
x_2^{(k)} \sim \mathcal{N}(x_2^{(k)} | x_2^{(k-1)}, \, \exp(\omega_2))
$$

```{admonition} Exercise 2
Using the equation above and your previous implementation, write a Python code that implements a hierarchical Gaussian random walk with the following parameters: $\omega_1 = -6.0$, $\omega_2 = -6.0$, $\mu_1 = 0.0$, $\mu_2 = -2.0$, $x_{1} = 0.0$ and $x_{2} = -2.0$
```

+++

### The continuous HGF

+++

In the following example, we illustrate how we can use the Hierarchical Gaussian Filter to filter and predict inputs in a continuous node.

```{code-cell} ipython3
# create a sime two-levels continuous HGF with defaults parameters
two_levels_continuous_hgf = HGF(
    n_levels=2,
    model_type="continuous",
    initial_mu={"1": 1.04, "2": 0.0},
    initial_pi={"1": 1e4, "2": 1e1},
    omega={"1": -8.0, "2": -1.0},
)
```

```{code-cell} ipython3
two_levels_continuous_hgf.plot_network()
```

```{code-cell} ipython3
# add new observations
two_levels_continuous_hgf = two_levels_continuous_hgf.input_data(input_data=timeseries)
```

```{code-cell} ipython3
two_levels_continuous_hgf.plot_trajectories();
```

```{code-cell} ipython3
two_levels_continuous_hgf.surprise()
```

```{admonition} Exercise 3
$\omega represents the tonic part of the variance (the part that is not affected by the parent node). Using the code example above, create another model with different values for $\omega$ at the second level. What is the consequence of changing this value on the beliefs trajectories? What is the "best" model in this context?
```

+++

### Parameters optimization

+++

In the final part of the exercise, you will be asked to apply the HGF to real world situation. For example, you can download wheather time series at the following link: https://renewables.ninja/. From there you can try:

* an agent has been living for 20 years in city A and has learn a model of weather that works well in this city, but decide to move and live in city B for one year. He is using the same model to understant weather changes in the new city. 


```{note}
The proposed exercise are more suggestions. You can use any time series that you find interesting, it should just come with a volatility that is interesting to model.
```

```{code-cell} ipython3
import pymc as pm
import numpy as np
from pyhgf.distribution import HGFDistribution
from pyhgf.response import first_level_gaussian_surprise

hgf_logp_op = HGFDistribution(
    n_levels=2,
    input_data=[timeseries],
    response_function=first_level_gaussian_surprise
)
```

```{code-cell} ipython3
with pm.Model() as two_level_hgf:

    # omegas priors
    omega_1 = pm.Uniform("omega_1", -20, -2.0)

    pm.Potential(
        "hgf_loglike",
        hgf_logp_op(
            omega_1=omega_1,
            omega_2=-2.0,
            omega_input=np.log(1e-4),
            rho_1=0.0,
            rho_2=0.0,
            pi_1=1e4,
            pi_2=1e1,
            mu_1=timeseries[0],
            mu_2=0.0,
            kappa_1=1.0,
            omega_3=np.nan,
            rho_3=np.nan,
            pi_3=np.nan,
            mu_3=np.nan,
            kappa_2=np.nan
        ),
    )
```

```{code-cell} ipython3
pm.model_to_graphviz(two_level_hgf)
```

```{code-cell} ipython3
with two_level_hgf:
    idata = pm.sample(chains=4)
```

```{code-cell} ipython3
az.plot_trace(idata)
```

```{code-cell} ipython3
az.summary(idata)
```

## Practice: Filtering the worlds weather to optimize behavior

+++

In the previous section, we introduced the basic computational concept behind the Hierarchical Gaussian Filter and illustrated:

1. How to fit the HGF to a time series.
2. How to find the parameters that optimize a simple response function.

In this section, we are going to apply this knowledge to more practical considerations and try to build an agent that can optimize its behaviour under volatile sensory inputs. We will take the example of an agent that experiences fluctuation in the weather and would like to optimize behaviour regarding whether or not he should carry an umbrella with him for the next day(s). Experiencing rain while not having an umbrella is extremely annoying for this agent (this elicits a lot of surprises), but carrying an umbrella in sunny weather is also annoying. You should therefore come up with a solution to help this agent optimise its decision. For the exercise we will consider that the agent cannot just look by the windows and check the weather, it has to make the decision one day for the next, using its current understanding of the weather.

We will use data from {citep}`pfenninger:2016, staffell:2016` that is made available at the following database: https://renewables.ninja/. This database contains hourly recordings of various weather parameters that have been tracked over one year at a different positions in the world. You can explore the database and use the recording you like, you can also compare agents trained in a different part of the globe. The procedure is the following: 
1. Set the point or search for a location
2. Extend the weather button and check all boxes (if you wish to use the premade processing script) or select your data structure.
3. Press "run" and wait for the simulation to run
4. Save hourly output as CSV

Alternatively, we provide here a data frame that can be easily loaded, it is the weather parameters recorded in Aarhus:

```{code-cell} ipython3
aarhus_weather_df = pd.read_csv("https://raw.githubusercontent.com/ilabcode/hgf-data/main/datasets/weather.csv")
aarhus_weather_df.head()
```

The data frame contains the following parameters, recorded every hour over the year of 2019:

- t2m: the 2-meter above ground level air temperature
- prectotland : The rain precipitation rate (mm/hour)
- precsnoland : Snow precipitation rate (mm/hour)
- snomas : Total snow storage land (kg/m2)
- rhoa : Air density at surface (kg/m3)
- swgdn : Surface incoming shortwave flux (W/m2) (considering cloud cover) (The value at the surface is approximately 1000 W/m2 on a clear day at solar noon in the summer months) 
- swtdn : Toa (top of atmosphere) incoming shortwave flux (W/m2)
- cldtot : Total cloud area fraction. An average over grid cells and summed over all height above ground ([0,1] scale where 0 is no cloud and 1 is very cloudy)

+++

```{admonition} Exercises
- Select a city and download a recording OR use the data frame loaded above.
- Choose which weather variables to work with.
- Set up an HGF structure and run it forward on the data.
- Set up an agent who has to decide something based on the weather (to carry an umbrella, or how many layers of clothes they have to put on based on temperature, cloudiness etc. ) and incorporate meaningful HGF parameters in the model.
- Compare the performances of two agents and interpret their differences. You have to choose what makes the two agents different: they can be trained on different data (e.g. from different cities, from different months...), have different HGF parameters, have different number of level, have different response functions (e.g. one agent can have pleasure carrying an umbrella when it is raining)... You can decide to optimize the parameters or not, this also depends on your hypothesis.
```

+++

To illustrate on possible workflow, we create an agent that tries to use the temperature to decide for taking an umbrella the next day. We use the first month of recording to train the agent:

```{code-cell} ipython3
# Load time series example data
timeserie = aarhus_weather_df["t2m"][:24*30]

# This is where we define all the model parameters - You can control the value of
# different variables at different levels using the corresponding dictionary.
hgf = HGF(
    n_levels=2,
    model_type="continuous",
    initial_mu={"1": timeserie[0], "2": .5},
    initial_pi={"1": 31e4, "2": 1e1},
    omega={"1":-6.0, "2": -3.0},
)

# add new observations
hgf.input_data(input_data=timeserie)

# visualization of the belief trajectories
hgf.plot_trajectories();
```

### Creating an agent 

Workflow of creating an agent with a response function:

- Get the trajectories from the hgf with the function `to_pandas()`.
- Determine which states could be interesting to work with, and how they might influence decision making.
- Set up a function taking the rows of the trajectories as inputs and return an action based on some comuptations you choose.

The most important state trajecotories of the HGF. 

- posterior mean: $\mu$
- posterior precision: $\pi$
- prediction mean: $\hat{\mu}$
- prediction precision:  $\hat{\pi}$
- surprise 

```{code-cell} ipython3
# get beliefs trajectories from the agent
trajectories = hgf.to_pandas()
trajectories.head()
```

```{code-cell} ipython3
# join the beliefs and observation
trajectories = pd.concat([aarhus_weather_df, trajectories], axis=1)
```

```{code-cell} ipython3
# 1 - create a decision function: based on the beliefs over the previous hour/day/week, decide if the agent should carry an umbrella
```

```{code-cell} ipython3
# 2 - create an outcome function: based on the parameters that were recorded in the main data fram, what was the outcome (raining or not?)
```

```{code-cell} ipython3
# 3  -compute the surprise: how surprise is the agent if it was raining and no umbrella etc...
```
