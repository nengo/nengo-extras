import numpy as np
from nengo.dists import get_samples
from nengo.neurons import Direct
from nengo.solvers import LstsqL2
from nengo.builder.connection import solve_for_decoders
from nengo.builder.ensemble import gen_eval_points


def tune_ens_parameters(ens, function=None, solver=None, rng=None, n=1000):
    """Find good ensemble parameters for decoding a particular function.

    Randomly generate many sets of parameters and determine the decoding error
    for each. Then set the ensemble parameters to those with the lowest
    decoding error. The "ensemble parameters" are the encoders, gains, biases,
    and evaluation points.

    Parameters
    ----------
    ens : Ensemble
        The ensemble to optimize.
    function : callable, optional
        The target function to optimize for. Defaults to the identity function.
    solver : nengo.solvers.Solver, optional
        The solver to use for finding the decoders. Default: ``LstsqL2()``
    rng : numpy.random.RandomState, optional
        The random number generator to use. Default: ``np.random``
    n : int, optional
        The number of random combinations to test. Default: 1000
    """

    if solver is None:
        solver = LstsqL2()
    if rng is None:
        rng = np.random
    if isinstance(ens.neuron_type, Direct):
        raise ValueError("Parameters do not apply to Direct mode ensembles")

    # use the same evaluation points for all trials
    eval_points = gen_eval_points(ens, ens.eval_points, rng=rng)
    targets = (np.array([function(ep) for ep in eval_points])
               if function is not None else eval_points)

    # --- try random parameters and record error
    errors = []
    for i in range(n):
        # --- generate random parameters
        if ens.gain is None and ens.bias is None:
            max_rates = get_samples(ens.max_rates, ens.n_neurons, rng=rng)
            intercepts = get_samples(ens.intercepts, ens.n_neurons, rng=rng)
            gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
        elif ens.gain is not None and ens.bias is not None:
            gain = get_samples(ens.gain, ens.n_neurons, rng=rng)
            bias = get_samples(ens.bias, ens.n_neurons, rng=rng)
        else:
            raise NotImplementedError("Mixed gain/bias and rates/ints")

        encoders = get_samples(
            ens.encoders, ens.n_neurons, ens.dimensions, rng=rng)

        # --- determine residual
        x = np.dot(eval_points, encoders.T / ens.radius)
        decoders, info = solve_for_decoders(
            solver, ens.neuron_type, gain, bias, x, targets, rng)
        error = info['rmses'].mean()

        errors.append((error, encoders, gain, bias, eval_points))

    # --- set parameters to those with the lowest error
    errors.sort(key=lambda x: x[0])
    ens.encoders, ens.gain, ens.bias, ens.eval_points = errors[0][1:]
