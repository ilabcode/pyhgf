# Belief updates in the HGF: Computations of nodes

In the approximate inversion of the generative model presented above, (Mathys, 2011) derived a set of simple, one-step update equations that represent changes in beliefs about the hidden states specified in the generative model. For each state, a belief is held (and updated for every new input) by the agent and described as a Gaussian distribution, fully characterized by its mean $\mu_i^{(k)}$ and its inverse variance, or precision, $\pi_i^{(k)}$ on a given trial $k$. We conceptualize each belief as a node in a network, where belief updates involve computations within nodes as well as message passing between nodes. The computations of any node within an experimental trial can be ordered in time as shown in the box:

> Node *i* at trial *k*
>
>(compute $\mathrm{prediction}^{(k)}_i$)  
>&larr; receive $\mathrm{PE}^{(k)}_{i-1}$ from $\mathrm{node}_{i-1}$
>
>UPDATE step  
>compute $\mathrm{posterior}^{(k)}_i$  
>*given:* $\mathrm{PE}^{(k)}_{i-1}$ and $\mathrm{prediction}^{(k)}_i$  
>&rarr; send $\mathrm{posterior}^{(k)}_i$ to $\mathrm{node}_{i-1}$
>
>PE step  
>compute $\mathrm{PE}^{(k)}_i$  
>*given:* $\mathrm{prediction}^{(k)}_i$ and $\mathrm{posterior}^{(k)}_i$  
>&rarr; send $\mathrm{PE}^{(k)}_i$ to $\mathrm{node}_{i+1}$  
>&larr; receive $\mathrm{posterior}^{(k)}_{i+1}$ from $\mathrm{node}_{i+1}$  
>
>PREDICTION step  
>compute $\mathrm{prediction}^{(k+1)}_i$  
>*given:* $\mathrm{posterior}^{(k)}_i$ and $\mathrm{posterior}^{(k)}_{i+1}$  

The exact computations in each step depend on the nature of the coupling (via **VAPE**s vs. **VOPE**s) with the parent and children nodes and will be outlined in the following two chapters.

Note that we have placed the **PREDICTION** step in the end of a trial. This is because usually, we think about the beginning of a trial as starting with receiving a new input, and of a prediction as being present before that input is received. However, in some variants of the HGF the prediction also depends on the time that has passed in between trials, which is something that can only be evaluated once the new input arrives - hence the additional computation of the (current) prediction in the beginning of the trial. Conceptually, it makes most sense to think of the prediction as happening continuously between trials. For implementational purposes, it is however most convenient to only compute the prediction once the new input (and with it its arrival time) enters. This ensures both that the posterior means of parent nodes have had enough time to be sent back to their children for preparation for the new input, and that the arrival time of the new input can be taken into account appropriately.
