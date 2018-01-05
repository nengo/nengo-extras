import numpy as np
import nengo
from surrogate import SurrogateEnsemble, Mapping
import matplotlib.pyplot as plt
import scipy
import math
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import arma_generate_sample
import time

##### TEST 1 #####

# N = 100
# D = 5

# ensemble_config = {
#     'n_neurons': N,
#     'dimensions': D
# }

# def function(x):
#     return x[0]*1 + x[1]*2 + x[2]*3 + x[3]*4 + x[4]*5

# lin_components = [
#     Mapping(lambda x: x[0]*1, 5, 1, [0], [0]),
#     Mapping(lambda x: x[1]*2, 5, 1, [1], [0]),
#     Mapping(lambda x: x[2]*3, 5, 1, [2], [0]),
#     Mapping(lambda x: x[3]*4, 5, 1, [3], [0]),
#     Mapping(lambda x: x[4]*5, 5, 1, [4], [0])
# ]

# connection = nengo.Connection(
#     nengo.Node([1,1,1,1,1], add_to_container=False),
#     nengo.Node(size_in=1, add_to_container=False),
#     synapse=0.005, function=function, add_to_container=False
# )

# def input_func(t):
#     return 0.2*math.sin(t)


##### TEST 2 #####

# N = 100
# D = 1

# ensemble_config = {
#     'n_neurons': N,
#     'dimensions': D
# }

# def function(x):
#     return x**2



# lin_components = [
#     Mapping(function, 1, 1, [0], [0]),
# ]

# connection = nengo.Connection(
#     nengo.Node([1], add_to_container=False),
#     nengo.Node(size_in=1, add_to_container=False),
#     synapse=0.005, function=function, add_to_container=False
# )

# def input_func(t):
#     return math.sin(t)


### TEST 3 #####

# N = 100
# D = 2

# ensemble_config = {
#     'n_neurons': N,
#     'dimensions': D
# }

# def function(x):
#     return x[0]*x[1]


# lin_components = [
#     Mapping(function, 2, 1, [0,1], [0]),
# ]

# connection = nengo.Connection(
#     nengo.Node([1,1], add_to_container=False),
#     nengo.Node(size_in=1, add_to_container=False),
#     synapse=0.005, function=function, add_to_container=False
# )

# def input_func(t):
#     return [-2 + t*4/6.0, 2 - t*4/6.0]
    # return [0.3*math.cos(t), 0.3*math.cos(t)]
    # return [-0.3928371, 0.54997194]



##### TEST 4 #####


# N = 100
# D = 5

# ensemble_config = {
#     'n_neurons': N,
#     'dimensions': D
# }

# def function(x):
#     return x[0]**2


# lin_components = [
#     Mapping(function, 5, 1, [0], [0]),
# ]

# connection = nengo.Connection(
#     nengo.Node([0,0,0,0,0], add_to_container=False),
#     nengo.Node(size_in=1, add_to_container=False),
#     synapse=0.005, function=function, add_to_container=False
# )

# def input_func(t):
#     return [0, 1 - t*2/6.0, 0, 1 - t*2/6.0, 0]


##### TEST 5 #####

# N = 100
# D = 3

# ensemble_config = {
#     'n_neurons': N,
#     'dimensions': D
# }

# def function(x):
#     return x[0]*1 + x[1]*2 + x[2]*3

# lin_components = [
#     Mapping(lambda x:  x[0]*1+x[1]*2+x[2]*3, , 1, [0,1,2], [0])
# ]

# connection = nengo.Connection(
#     nengo.Node([1,1,1], add_to_container=False),
#     nengo.Node(size_in=1, add_to_container=False),
#     synapse=0.005, function=function, add_to_container=False
# )

# def input_func(t):
#     return math.sin(t)



#### TEST 6 #####

ensemble_config = {
    'n_neurons': 300,
    'dimensions': 3
}

def function(x):
    return x[0]**3

components = [
    Mapping(lambda x: x[0]**3, 3, 1, [0], [0])
]

connection = nengo.Connection(
    nengo.Node([1,1,1], add_to_container=False),
    nengo.Node(size_in=1, add_to_container=False),
    synapse=0.005, function=function, add_to_container=False
)

def input_func(t):
    return [ -1.5 + 3*t/6, 0, 1.5 - 3*t/6]


surrogate = SurrogateEnsemble(ensemble_config, connection, components, dt=0.001)
import time
start = time.time()
sim, ens, out_conn, p_in, p_out = surrogate.build_network(
    input_func, surrogate.connection, surrogate.connection.function,
    surrogate.connection.transform, surrogate.dt, 6
)
end = time.time()
full_simulation_build_time = end-start
print "==================="
print "full simulation build time: ", full_simulation_build_time
print "==================="

start = time.time()
surrogate.build(noise_sampling_steps=3, seed=6)
end = time.time()
population_mode_build_time = end-start
print "==================="
print "population mode build time:", population_mode_build_time
print "==================="


for i in [3,5,7,20]:
    print "\n==================="
    print "Simulation length = %d seconds" % i
    print "==================="

    start = time.time()
    sim.run(i)
    end = time.time()
    full_simulation_length = end - start
    print "\nfull simulation took: ", full_simulation_length
    print "Overall (build+simulation) time took: ", full_simulation_length + full_simulation_build_time


    out_shape = sim.data[p_out].shape
    start = time.time()
    surrogate.emulate(sim.data[p_in], out_shape)
    end = time.time()
    poplation_mode_simulation_length = end - start
    print "\npopulation mode simulation took: ", poplation_mode_simulation_length
    print "Overall (build+simulation) time took: ", poplation_mode_simulation_length + population_mode_build_time



# surrogate = SurrogateEnsemble(ensemble_config, connection, lin_components, dt=0.001)
# surrogate.build(noise_sampling_steps=3, seed=6)
# surrogate.test_performance(input_func, length=6, seed=6)

# plt.figure(figsize=(12,6))
# plt.title("uniformly sampled bias at 100 points of input x")
# plt.xlabel('input x')
# plt.ylabel('bias values')
# plt.plot(surrogate.bias_eval_points[:,0], surrogate.sampled_bias[:,0], 'g', label="bias", alpha=0.7)
# plt.legend()

# plt.figure(figsize=(12,6))
# plt.title("sampled noise from a 0.5 second simulation")
# plt.xlabel('time t')
# plt.ylabel('noise values')
# plt.plot(surrogate.training_tranges[0], surrogate.training_sim_noises[0][:,0], 'g', label="noise", alpha=0.7)
# plt.legend()







# plt.figure(figsize=(12,6))
# plt.title("sampled noises")
# plt.plot(surrogate.training_tranges[0], surrogate.training_sim_noises[0][:,0], 'g', label="Noise 0", alpha=0.5)
# plt.plot(surrogate.training_tranges[1], surrogate.training_sim_noises[1][:,0], 'b', label="Noise 1", alpha=0.5)
# plt.plot(surrogate.training_tranges[2], surrogate.training_sim_noises[2][:,0], 'r', label="Noise 2", alpha=0.5)
# plt.legend()


plt.figure(figsize=(12,6))
plt.title('Comparison of Real and Modeled Noises')
plt.xlabel('time t')
plt.ylabel('bias values')
plt.plot(surrogate.trange, surrogate.sim_noise_values[:,0], 'r', label="Real Noise", alpha=0.7)
plt.plot(surrogate.trange, surrogate.est_noise_values[:,0], 'b', label="Modeled Noise", alpha=0.7)
plt.legend()

# plt.figure(figsize=(12,6))
# plt.title('Input Over Time')
# plt.xlabel('time t')
# plt.ylabel('input')
# plt.plot(surrogate.trange, surrogate.input[:,0], 'g', label="Input", alpha=0.7)
# plt.legend()

# plt.figure(figsize=(12,6))
# plt.title('Comparison of Bias Estimations')
# plt.xlabel('time t')
# plt.ylabel('bias values')
# plt.plot(surrogate.trange, surrogate.sim_bias_values[:,0], 'r', label="actual bias", alpha=0.7)
# plt.plot(surrogate.trange, surrogate.est_bias_values[:,0], 'b', label="estimated bias", alpha=0.7)
# plt.legend()



# def RMSE(y_hat_list, y_list):
#     assert len(y_hat_list) == len(y_list)
#     n = len(y_hat_list)

#     square_sum = 0
#     for i in range(n):
#         square_sum += math.pow(y_hat_list[i] - y_list[i], 2)

#     return math.sqrt(square_sum/float(n))



# plt.figure(figsize=(12,6))
# plt.title('PSD Analysis of Real and Modeled Noises')
# plt.xlabel('frequency')
# plt.ylabel('component of given frequency')
# plt.plot(surrogate.spike_freqs[0], surrogate.spike_PSDs[0], 'r', label="Real Noise Power Spectra", alpha=0.7)
# plt.plot(surrogate.model_freqs[0], surrogate.model_PSDs[0], 'b', label="Model Noise Power Spectra", alpha=0.7)
# plt.legend()


# plt.figure(figsize=(12,6))
# plt.title("Comparison of Outputs")
# plt.plot(surrogate.trange, surrogate.ideal_output[:,0], 'g', label="ideal output", alpha=0.7)
# plt.plot(surrogate.trange, surrogate.actual_output[:,0], 'r', label="ideal output", alpha=0.7)
# plt.plot(surrogate.trange, surrogate.surrogate_output[:,0], 'b', label="surrogate output", alpha=0.7)
# plt.legend()






plt.show()


