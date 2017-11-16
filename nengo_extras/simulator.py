"""
Utils for helping with Simulators
"""
from nengo.utils.compat import iteritems


class State(object):
    """Wrapper for saving and loading simulators.
    """

    def __init__(self, sim):
        self.sim = sim

    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        state = dict(state)
        Simulator = state.pop('Simulator')
        model = state.pop('model')
        sim = Simulator(network=None, model=model)
        self.__init__(sim)
        self.set_state(state)

    def get_state(self, probes=False):
        if probes:
            raise NotImplementedError("Cannot save probes yet")

        Simulator = self.sim.__class__
        model = self.sim.model
        signals = {
            k: v for k, v in iteritems(self.sim.signals) if not k.readonly}
        return dict(Simulator=Simulator, model=model, signals=signals)

    def set_state(self, state):
        state = dict(state)
        signals = state.pop('signals')
        for k, v in iteritems(signals):
            self.sim.signals[k] = v
        assert len(state) == 0
        self.sim._probe_step_time()

    # @classmethod
    # def load(cls, state):
    #     state = dict(state)
    #     Simulator = state.pop('Simulator')
    #     model = state.pop('model')
    #     sim = Simulator(network=None, model=model)
    #     wrapper = cls(sim)
    #     wrapper.set_state(state)
    #     return wrapper
