# import dill
import matplotlib.pyplot as plt
import nengo
import numpy as np

from nengo.utils.compat import pickle
from nengo_extras.simulator import State

# filename = 'test_save.dil'

with nengo.Network() as model:
    u = nengo.Node(np.sin)
    a = nengo.Ensemble(100, 1)
    b = nengo.Ensemble(100, 1)
    c = nengo.Connection(a, b)
    bp = nengo.Probe(b)

with nengo.Simulator(model) as sim:
    sim.run(3.0)
    # print(type(sim.time))
    # print(sim.data['time'])

    plt.figure(1)
    plt.plot(sim.trange(), sim.data[bp])

    state = State(sim)
    pkls = pickle.dumps(dict(state=state, bp=bp))

del model, sim, bp

pkl = pickle.loads(pkls)
state = pkl['state']
sim = state.sim
bp = pkl['bp']

with sim:
    sim.run(3.0)

    plt.figure(2)
    plt.plot(sim.trange(), sim.data[bp])

plt.show()
