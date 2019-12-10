import numpy as np
import scipy.special
import nengo

class AreaIntercepts(nengo.dists.Distribution):
    """ Generate an optimally distributed set of intercepts in
    high-dimensional space.
    """
    dimensions = nengo.params.NumberParam('dimensions')
    base = nengo.dists.DistributionParam('base')

    def __init__(self, dimensions, base=nengo.dists.Uniform(-1, 1)):
        super(AreaIntercepts, self).__init__()
        self.dimensions = dimensions
        self.base = base

    def __repr(self):
        return ("AreaIntercepts(dimensions=%r, base=%r)" %
                (self.dimensions, self.base))

    def transform(self, x):
        sign = 1
        if x > 0:
            x = -x
            sign = -1
        return sign * np.sqrt(1 - scipy.special.betaincinv(
            (self.dimensions + 1) / 2.0, 0.5, x + 1))

    def sample(self, n, d=None, rng=np.random):
        s = self.base.sample(n=n, d=d, rng=rng)
        for ii, ss in enumerate(s):
            s[ii] = self.transform(ss)
        return s


class Triangular(nengo.dists.Distribution):
    """ Generate an optimally distributed set of intercepts in
    high-dimensional space using a triangular distribution.
    """
    left = nengo.params.NumberParam('dimensions')
    right = nengo.params.NumberParam('dimensions')
    mode = nengo.params.NumberParam('dimensions')

    def __init__(self, left, mode, right):
        super(Triangular, self).__init__()
        self.left = left
        self.right = right
        self.mode = mode

    def __repr__(self):
        return ("Triangular(left=%r, mode=%r, right=%r)" %
                (self.left, self.mode, self.right))

    def sample(self, n, d=None, rng=np.random):
        # the distribution is the mirror of the user mode/bounds
        # flip the output here so it matches the user input
        # (ie mode=0.4 will be at 0.4 instead of -0.4)
        if d is None:
            return -1*rng.triangular(self.left, self.mode, self.right, size=n)
        else:
            return -1*rng.triangular(
                self.left, self.mode, self.right, size=(n, d))

def generate(n_input, n_ensembles, n_neurons, bounds, mode, seed=0):
    """ Returns an array in the shape of (n_ensembles, n_neurons)
    of intercepts optimally distrubuted in high-dimensional space
    using a triangular distribution

    Parameters
    ----------
    n_input: int
        the number of input dimensions
    n_ensembles: int
        the number of ensembles to generate intercepts for
    n_neurons: int
        the number of neurons in each ensemble
        NOTE: each ensembles must contain the same number of neurons
    bounds: list of two floats
        the lower and upper bounds for the intercepts
    mode: float
        the the desired mode for the triangular distribution
    seed: int, Optional (Default: 0)
        the seed for the numpy rng
    """
    np.random.seed = seed
    intercepts_dist = AreaIntercepts(
        dimensions=n_input, base=Triangular(
            bounds[0], mode, bounds[1])
        )
    intercepts = intercepts_dist.sample(n=n_neurons*n_ensembles)
    intercepts = intercepts.reshape(n_ensembles, n_neurons)

    return intercepts
