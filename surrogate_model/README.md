## Surrogate Model ##

Please refer to [the Jupytar notebook](https://github.com/nengo/nengo_extras/blob/surrogate_model/surrogate_model/surrogate_model_overview.ipynb) for a full technical overview of the the surrogate model.

### Brief Summary ###

The surrogate model is meant to achieve what we refer to as the *population mode*. We want to emulate the output of a population of NEF neurons without simulating the encoding and decoding processes for each neuron. The goal is to save computational cost while maitaining accuracy with respect to a full NEF simulation. 

Roughly, the surrogate model breaks the NEF population output into a low frequency component (refered to as `bias`) and high frequency component (refered to as `noise`). We then characterize each component individually using various models. Currently, the `bias` term is fit using polynomial interpolations, and the `noise` term is fit using [ARMA model](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model).

Once the models are fit to the population output, we can then run the models instead of full simulation.

Results on toy networks are presented in [the Jupytar notebook](https://github.com/nengo/nengo_extras/blob/surrogate_model/surrogate_model/surrogate_model_overview.ipynb). The interpolation methods have a hard time modeling the `bias` term when the population encodes a high dimension.

### Folders ###

The `legacy` folder contains previous iterations. There were major structual changes, but some logic in the current iteration was inspired from the previous iterations, and so they are kept for reference. 
