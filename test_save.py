import dill
import matplotlib.pyplot as plt
import nengo
import numpy as np

from nengo_extras.simulator import State

filename = 'test_save.dil'

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
    plt.plot(sim.data['time'], sim.data[bp])

    state = State(sim)
    with open(filename, 'wb') as fh:
        dill.dump(state, fh)

del model
del sim

with open(filename, 'rb') as fh:
    state = dill.load(fh)
    # state = State.load(data)
    sim = state.sim

with sim:
    sim.run(3.0)

    plt.figure(2)
    plt.plot(sim.data['time'], sim.data[bp])

plt.show()
