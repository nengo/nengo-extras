import numpy as np
import nengo
import nengo_gui.ipython
import matplotlib.pyplot as plt
import nengo.utils.numpy as npext
from SurrogateModel_ver3 import SurrogateModel


# Variables initialization
tau_syn = 0.005
sim_time = 4
dt = 0.001
rad = 30
dim = 1
ARMA_orders = [2, 0, 2]
trange = np.linspace(0, sim_time, num=sim_time/dt)

seed = None


# Declare the input vector
input_vec = (lambda t: t)
# input_vec = (lambda t: t*0+1)
# input_vec = (lambda t: np.cos(t))
# input_vec = (lambda t: t**2)
# input_vec = (lambda t: np.log(t))


N = 50
D = 1
L = 1

# seed 2,3
model = nengo.Network(label="Communications Channel", seed=2)
with model:
    input = nengo.Node(input_vec)
    output = nengo.Node(None, size_in=D) # For decoders

    layers = [nengo.Ensemble(N, D, radius=rad) for i in range(L)]
    nengo.Connection(input, layers[0], synapse=tau_syn)
    for i in range(L-1):
        nengo.Connection(layers[i], layers[i+1], synapse=tau_syn)
    nengo.Connection(layers[-1], output)

    pInput = nengo.Probe(input, 'output')
    pOutput = nengo.Probe(layers[-1], 'decoded_output', synapse=tau_syn)


print "asdsad"



# Pass in the model to the SurrogateModel API; initialize default parameters on the API side
surr_model = SurrogateModel(model, dim) # input of: ramp, const, cos, squared, log

# Simulate by model.run
surr_model.run(dt, sim_time)
surr_model.estimateSurrModel(ARMA_orders, tau_syn)


# Plot the surrogate model estimation
trange = surr_model.trange
print "surr_model.RMSE: {}".format(surr_model.RMSE)

plt.figure(figsize=(12,6))
plt.plot(trange, surr_model.bias_values_sim.T, 'r', label = "Bias sim")
plt.plot(trange, surr_model.bias_values_est.T, 'g', label= "Bias est")
plt.legend()

plt.figure(figsize=(12,6))
# plt.plot(trange, surr_model.noise_values_sim.T, 'r', label ="Noise sim")
plt.plot(trange, surr_model.noise_values_filt.T, 'g', label="Noise filt")
plt.plot(trange, surr_model.noise_values_est.T, 'b', label="Noise est")
plt.legend()

plt.figure(figsize=(12,6))
# plt.plot(trange, surr_model.input.T, 'k', label='Input')
# plt.plot(trange, surr_model.output.T, 'r', label = 'Actual output')
# # plt.plot(trange, surr_model.static_output.T, 'g', label = 'Static output' )
# plt.plot(trange, surr_model.surr_output.T, 'b', label="Surrogate output")
plt.legend()
plt.show()
# print "surr_model.RMSE: {}".format(surr_model.RMSE)

