---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(theory)=
# The Hierarchical Gaussian Filter
In this notebook, we introduce the main concepts on which the Hierarchical Gaussian Filter (HGF) is based. We describe the main equations and illustrate the examples with Python code. 

This part of the documentation is work in progress. For now, please refer to the following publications for the theory behind the models implemented in this toolbox:

* Introducing the generative model for volatility coupling and the corresponding belief update equations: <cite>[Mathys et al. 2011][1]</cite> and <cite>[Mathys et al. 2014][2]</cite>
* Introducing the generative model for value coupling and the corresponding belief update equations, as well as the conceptualisation of the belief updates as a network of interacting (belief) nodes: chapter 4 in <cite>[Weber 2020][3]</cite>

[1]: https://doi.org/10.3389/fnhum.2011.00039
[2]: https://doi.org/10.3389/fnhum.2014.00825
[3]: https://doi.org/10.3929/ethz-b-000476505

