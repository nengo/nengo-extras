import numpy as np
import nengo
import nengo_gui.ipython
import matplotlib.pyplot as plt
from SurrogateEnsemble import SurrogateEnsemble
import scipy
import math
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import arma_generate_sample

N = 100
D = 1
# tau_syn = 0.005
# dt = 0.001

ensemble_config = {
    'n_neurons': N,
    'dimensions': D
}

def function(x):

    return (np.cos(x))**2

def input_func(t):
    return 0

surrogate = SurrogateEnsemble(ensemble_config, function, synapse=0.005, dt=0.001)
surrogate.build(length=2, seed=6)
surrogate.test_performance(input_func)

# plt.figure(figsize=(12,6))
# plt.plot(surrogate.training_trange, surrogate.ens_input, 'b', label = "input")
# # plt.plot(surrogate.training_trange, surrogate.ens_output, 'r', label = "actual")
# # # plt.plot(surrogate.training_trange, surrogate.training_sim_static_output, 'b', label = "static")
# # plt.plot(surrogate.training_trange, surrogate.training_ideal_output, 'g', label= "ideal")
# plt.legend()



# plt.figure(figsize=(12,6))
# plt.plot(surrogate.trange, surrogate.est_bias_values, 'r', label = "Bias est")
# plt.plot(surrogate.trange, surrogate.sim_bias_values, 'g', label= "Bias sim")
# plt.legend()





# training_est_noises = arma_generate_sample(
#     ar=np.r_[1, -surrogate.noise_model.arparams],
#     ma=np.r_[1, surrogate.noise_model.maparams],
#     nsample=len(surrogate.training_filt_sim_noise),
#     sigma=surrogate.noise_model.sigma2**0.5
# ).reshape(surrogate.training_filt_sim_noise.shape)

# plt.figure(figsize=(12,6))
# # plt.plot(surrogate.training_trange, surrogate.training_sim_noise, 'r', label="Noise sim")
# plt.plot(surrogate.training_trange, surrogate.training_filt_sim_noise, 'g', label="Noise filt")
# plt.plot(surrogate.training_trange, surrogate.training_filt_sim_noise_probed, 'y', label="Noise filt probed")
# # plt.plot(surrogate.training_trange, training_est_noises, 'b', label="Noise est")
# plt.legend()


# filt_freq, filt_PSD = signal.periodogram(surrogate.training_filt_sim_noise.ravel(),fs=1/surrogate.dt,window='hanning')
# model_freq, model_PSD = signal.periodogram(training_est_noises.ravel(),fs=1/surrogate.dt,window='hanning')


# plt.figure(figsize=(12,6))
# plt.plot(filt_freq, filt_PSD, 'g', label="real power spectra")
# plt.plot(model_freq, model_PSD, 'b', label="model power spectra")
# plt.legend()





plt.figure(figsize=(12,6))
plt.plot(surrogate.trange, surrogate.filt_noise_values, 'g', label="Noise filt")
plt.plot(surrogate.trange, surrogate.est_noise_values, 'b', label="Noise est")
plt.legend()

print surrogate.filt_PSD
print surrogate.model_PSD
filt_PSD_smoothed = scipy.stats.gaussian_kde(surrogate.filt_PSD*1e11,  bw_method=1e-6).evaluate(surrogate.filt_freq)
model_PSD_smoothed = scipy.stats.gaussian_kde(surrogate.model_PSD*1e11, bw_method=1e-6).evaluate(surrogate.model_freq)

# print filt_PSD_smoothed

# import ipdb; ipdb.set_trace()

plt.figure(figsize=(12,6))
plt.plot(surrogate.filt_freq, surrogate.filt_PSD, 'r', label="real power spectra")
plt.plot(surrogate.model_freq, surrogate.model_PSD, 'y', label="model power spectra")
plt.legend()

plt.figure(figsize=(12,6))
# plt.hist(surrogate.filt_PSD, bins=100)
# plt.hist(surrogate.model_PSD, bins=100)
# plt.plot(surrogate.filt_freq, surrogate.filt_PSD, 'r', label="real power spectra")
# plt.plot(surrogate.model_freq, surrogate.model_PSD, 'y', label="model power spectra")
plt.plot(surrogate.filt_freq, filt_PSD_smoothed, 'g', label="real power spectra")
plt.plot(surrogate.model_freq, model_PSD_smoothed, 'b', label="model power spectra")
plt.legend()

plt.figure(figsize=(12,6))
plt.plot(surrogate.trange, surrogate.ideal_output, 'g', label="ideal output")
plt.plot(surrogate.trange, surrogate.surrogate_output, 'r', label="surrogate output", alpha=0.7)
plt.plot(surrogate.trange, surrogate.filtered_actual_output, 'b', label="actual_output", alpha=0.7)
plt.legend()



# plt.figure(figsize=(12,6))
# # plt.plot(trange, surr_model.input.T, 'k', label='Input')
# plt.plot(surroage.trange, surr_model.output.T, 'r', label = 'Actual output')
# # plt.plot(trange, surr_model.static_output.T, 'g', label = 'Static output' )
# plt.plot(surroage.trange, surr_model.surr_output.T, 'b', label="Surrogate output")
# plt.legend()


plt.show()


