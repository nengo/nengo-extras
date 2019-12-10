import pytest

import numpy as np
import matplotlib.pyplot as plt

from nengo_extras import triangular_intercepts

@pytest.mark.parametrize(
    ('n_input, n_ensembles, n_neurons'),(
    (1, 1, 100000),
    (1, 2, 10000)
    )
)
@pytest.mark.parametrize(
    ('bounds, mode'),(
    ([-1, 1], 0),
    ([-1, 1], 0.4),
    ([-0.3, 0.8], 0.4)
    )
)
def test_generate(plt, n_input, n_ensembles, n_neurons, bounds, mode):
    intercepts = triangular_intercepts.generate(
        n_input=n_input,
        n_ensembles=n_ensembles,
        n_neurons=n_neurons,
        bounds=bounds,
        mode=mode)

    assert intercepts.shape[0] == n_ensembles
    assert intercepts.shape[1] == n_neurons
    assert (np.asarray(intercepts)>=bounds[0]).all()
    assert (np.asarray(intercepts)<=bounds[1]).all()

    # round to one decimal to speed things up
    # equivalent to having a bin size of 0.1
    n_decimals = 1
    intercepts = np.around(intercepts, n_decimals)
    bin_size = 1/10**n_decimals
    # get a count of the unique intercepts
    for ii in range(n_ensembles):
        vals = np.unique(intercepts[ii])
        plt.figure()
        data = []
        for val in vals:
            count = list(intercepts[ii]).count(val)
            data.append({'val': val, 'count': count})
            plt.scatter(val, count)

        # generate line plot of the expected triangular shape
        # left side
        x1 = np.linspace(bounds[0], mode, len(vals))
        # right side
        x2 = np.linspace(mode, bounds[1], len(vals))
        # max count (should be at mode)
        max_int = bin_size*n_neurons/(0.5*(bounds[1]-bounds[0]))
        # line equations
        y1 = lambda x: (max_int/(mode-bounds[0]))*(x-bounds[0])
        y2 = lambda x: -(max_int/(bounds[1]-mode))*(x-bounds[1])
        plt.plot(x1, y1(x1), 'k', label='expected')
        plt.plot(x2, y2(x2), 'k')
        plt.legend()

        tolerance = 0.01*n_neurons
        for entry in data:
            if entry['val'] < mode:
                y = y1
            else:
                y = y2
            try:
                assert(np.isclose(entry['count'], y(entry['val']),
                    atol=tolerance, rtol=0))
            except AssertionError as e:
                print(
                    'Intercept %f had a count of %f'
                    % (entry['val'], entry['count'])
                    + '\nExcepted: %f +/- %f\nDifference: %f'
                    % (y(entry['val']), tolerance,
                    (entry['count'] - y(entry['val']))))
                raise e


if __name__ == '__main__':
    test_generate()
