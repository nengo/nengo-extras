import time
import timeit

import nengo
import numpy as np
import pytest

from nengo_extras.simulators import RealTimeSimulator


def test_real_time_simulator(rng):
    with nengo.Network() as net:

        def rnd_delay(_):
            delay = rng.uniform(0, 5e-4)
            start = timeit.default_timer()
            while timeit.default_timer() - start < delay:
                pass
            return [0.0]

        nengo.Node(rnd_delay)

    with RealTimeSimulator(net) as sim:
        for _ in range(2):
            t_start = timeit.default_timer()
            sim.run(1)
            elapsed = timeit.default_timer() - t_start

            assert np.allclose(elapsed, 1, atol=0.02)

            # verify that resetting can be used to address external delays
            time.sleep(0.1)
            sim.reset()

    # warning when running slower than real-time
    with nengo.Network() as net:

        def long_delay(_):
            time.sleep(0.1)

        nengo.Node(long_delay)

    with RealTimeSimulator(net) as sim:
        with pytest.warns(UserWarning, match="RealTimeSimulator is running"):
            sim.step()
