---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(example_1)=
# Example 2: Volatility coupling with an input node

```{code-cell} ipython3
%%capture
import sys
if 'google.colab' in sys.modules:
    ! pip install pyhgf
```

```{code-cell} ipython3
from pyhgf.distribution import HGFDistribution
from pyhgf.model import HGF
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
```

Where the standard continuous HGF assumes a known precision in the input node (usually set to something high), this assumption can be relaxed and the filter can also try to estimate this quantity from the data.

```{code-cell} ipython3
input_data = np.random.normal(size=1000)
```

```{code-cell} ipython3
jget_hgf = (
    HGF(model_type=None)
    .add_input_node(kind="continuous")
    .add_value_parent(children_idxs=[0])
    .add_volatility_parent(children_idxs=[0])
    .add_volatility_parent(children_idxs=[1])
    .init()
)
jget_hgf.plot_network()
```

```{code-cell} ipython3
jget_hgf.attributes
```

```{code-cell} ipython3
jget_hgf.input_data(input_data[:30])
```

```{code-cell} ipython3
jget_hgf.plot_trajectories()
```

```{code-cell} ipython3
jget_hgf.to_pandas()
```

```{code-cell} ipython3

```
