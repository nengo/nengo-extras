import time
import timeit
import warnings

import nengo


class RealTimeSimulator(nengo.Simulator):
    """
    A simulator that will not run faster than real time.

    Nothing prevents this simulator from running slower than real time
    (although it will print a warning if that is the case).

    In addition, this Simulator measures time from the first call to
    ``sim.step``/``sim.run``, so adding some large gap in the middle of a
    run will throw off the timing.  Calling ``sim.reset()`` will reset the
    start time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_time = None

    def step(self, *args, **kwargs):
        if self.start_time is None:
            self.start_time = timeit.default_timer()

        super().step(*args, **kwargs)

        elapsed = timeit.default_timer() - self.start_time

        if elapsed < self.time:
            time.sleep(self.time - elapsed)
        elif elapsed - self.time > 0.05:
            warnings.warn("RealTimeSimulator is running slower than real time")

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

        self.start_time = None
