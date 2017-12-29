# Surrogate API that compares the performance between the existing Nengo model
import random
import nengo
import nengo.utils.numpy as npext

import numpy as np
# import pandas as pd

from nengo.utils.ensemble import tuning_curves
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from scipy import signal
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import arma_generate_sample

class SurrogateModel(object):
    def __init__(self, model, dim, seed=None):
        self.dim = dim                           # model dimension

        self.stim_probe = model.probes[-1]
        self.pop_probe = model.probes[-1]

        self.dt = None                           # time step
        self.sim = None                          # simulator instance
        self.trange = []
        self.output = []                         # last ensemble's output
        self.input = []                          # neural input

        self.rates = []                          # last ensemble's rate
        self.decoders = []                       # last ensemble's decoder
        self.static_output = []                  # last ensemble's static_output

        # Params for bias model
        self.bias_values_sim = []
        self.bias_values_est = []

        # Params for noise model
        self.noise_values_sim = []
        self.noise_values_filt = []
        self.noise_values_est = []

        self.spike_freq = []
        self.spike_PSD = []
        self.filt_freq = []
        self.filt_PSD = []
        self.model_freq = []
        self.model_PSD = []

        self.RMSE = []

        self.surr_output = []

    #############################
    # SURROGATE MODEL FUNCITONS #
    #############################

    def run(self, dt, sim_time):
        """
        Run simulation with the surrogate model

        Params:
        dt: timestep
        sim_time: total simulated time
        """
        self.dt = dt
        self.sim_time = sim_time

        with nengo.Simulator(self.model, dt) as sim:
            sim.run(sim_time)

        self.sim = sim
        self.trange = sim.trange()

        self.input = sim.data[self.stim_probe].T
        self.output = sim.data[self.pop_probe].T

        pop = self.model.ensembles[-1]
        conn_end = self.model.connections[-1]
        eval_points, rates = tuning_curves(pop, sim, inputs=self.input.T)
        print rates
        self.rates = rates.T
        self.decoders = sim.data[conn_end].weights
        self.static_output = np.dot(self.decoders, self.rates)

    def estimateSurrModel(self, ARMA_orders=[2,0,2], tau_syn=0.005):
        """
        Calculates the surrogate model based on the estimated noise and bias terms;
        the surrogate model has been already simulated in order to run this function

        NOTE: surr_output = input + bias_est + noise_est

        Steps:
        1. Ensure that the model has been simulated
        2. Estimate the bias and noise term
        3. Calculate the surr_model output
        5. Calculate RMSE in between the actual output and
        """
        if self.sim is None:
            raise ValueError("Surrogate model has not been simulated")

        self.createBiasModel()
        self.createNoiseModel(ARMA_orders, tau_syn)
        surr_output = (self.input          +
                      self.bias_values_est +
                      self.noise_values_est )

        self.surr_output = surr_output
        self.RMSE = np.sqrt((surr_output - self.output) ** 2).mean()


    ########################
    # BIAS MODEL FUNCTIONS #
    ########################

    def createBiasModel(self, mode="lin_interp"):
        """
        Creates bias model for estimating the bias term of the surrogate model.
        mode: different mode of bias approximation (types: lin_interp, poly_reg, fourier_reg)

        Steps:
        1. Calculate the simulated bias
        2. Estimate the bias term based on the simulated bias

        Params:
        mode: interpolation method
        """
        if self.dim == 1:
            self.calcSimBias()
            self.estimateBias(mode)

        # TODO: Finish implementing for multidimensional case

    def calcSimBias(self):
        """
        Obtains samples of bias (distortion) error which can then later be fit to a
        model. Returns an array of bias errors at each eval points for each
        origin and an ideal values (ideal). (Note: actual = ideal + bias)
        """
        bias = self.static_output - self.input
        # print "static_output: {}".format(self.static_output.shape)
        # print "input: {}".format(self.input.shape)
        # print "bias shape: {}".format(bias.shape)
        self.bias_values_sim = bias

    def estimateBias(self, mode):
        """
        Estimates the simulated bias by interpolation method specified by "mode"

        Steps:
        1. Calculate the interpolation function using the exiting eval_points and the simulated bias term
        2. Generate new points based on the eval_points
        3. Calculate the estimated bias using the interpolation function from Step 1

        """
        eval_points = self.genBiasEvalPoints(dist="gaussian")
        eval_points = np.sort(eval_points)
        bias = self.bias_values_sim

        if self.dim == 1:
            eval_points = eval_points.squeeze()
            eval_points = np.sort(eval_points)
            # TODO: Complete different interpolation methods
            if mode == "lin_interp":
                func = interp1d(eval_points, bias)
                new_points = self.genBiasEvalPoints(radius=min(abs(eval_points[0]),
                                abs(eval_points[-1])),dist="gaussian")
                new_points = np.sort(new_points)
                self.bias_values_est = func(new_points).squeeze()


    def genBiasEvalPoints(self, radius=None, dist="uniform", mean=0, sd=1):
        """
        Generates bias evaluation points, based on the distribution within
        the range of [-1, 1] * radius.

        Params:
        radius: the representational radius of the ensemble.
        dist: distribution of eval_points; default is uniform distribution
        mean: mean of eval_points when dist is gaussian
        sd: standard deviation of eval_points when dist is gaussian
        """
        if radius == None:
            radius = self.model.ensembles[-1].radius

        if dist == "uniform":
            points = np.linspace(-1*radius, 1*radius, num=self.trange[-1]/self.dt)
            points = points.reshape((len(points),1))

            if self.dim > 1:
                points = np.asarray(npext.meshgrid_nd(*(self.dim * [points]))).T

        elif dist == "gaussian":
            # TODO: Implement generating samples under gaussian distribution for higher dimensions

            points = np.linspace(-1*radius, 1*radius, num=self.trange[-1]/self.dt)
            points = points.reshape((len(points),1))

            dimensions = self.static_output.shape
            np.random.seed()
            trunc_norm = truncnorm((points[0]-mean)/sd, (points[-1]-mean)/sd, loc=mean, scale=sd)
            points = sorted(trunc_norm.rvs(dimensions))
            points = np.asarray(points)

        return points

    #########################
    # NOISE MODEL FUNCTIONS #
    #########################

    def createNoiseModel(self, ARMA_orders, tau_syn):
        """
        Create noise model for estimating the noise term of the surrogate model

        Steps:
        1. Calculate the noise, the time variant variation and its power spectra density(PSD)
        2. Normalize the noise by passing it to the LPF and calculate its PSD
        3. Estimate the normalized noise using ARMA model with normalized randomly distributed points
        4. Calculate the PSD of the modeled noise

        Params:
        ARMA_orders: ARIMA model param; default is [2,0,2]
        tau_syn: synapse constant; default is 0.005
        """
        self.noise_values_sim = self.output - self.static_output
        self.noise_values_filt = self.filtNoise(self.noise_values_sim, tau_syn)

        self.spike_freq, self.spike_PSD = signal.periodogram(self.noise_values_sim, fs=1/self.dt, window='hanning')
        self.filt_freq, self.filt_PSD = signal.periodogram(self.noise_values_filt,fs=1/self.dt,window='hanning')

        self.noise_values_est = self.estimateNoise(self.noise_values_filt, ARMA_orders, tau_syn)
        self.model_freq, self.model_PSD = signal.periodogram(self.noise_values_est,fs=1/self.dt,window='hanning')

        return model_noise


    def filtNoise(self, noise, tau_syn):
        """
        Perform LPF on the simulated noise for filtering & shaping the data
        in gaussian distribution
        """
        sys_PSC = signal.TransferFunction(1, [tau_syn, 1])
        for d in range(self.dim):
            filt_noise_d = signal.lsim(sys_PSC, noise[d], self.trange)[1] # interested only in yout
            if d == 0:
                filt_noise = filt_noise_d
            else:
                filt_noise = np.vstack((filt_noise, filt_noise_d))

        return filt_noise.reshape((self.dim,len(filt_noise)))

    def estimateNoise(self, filt_noise, ARMA_orders, tau_syn):
        for d in range(self.dim):
            model = SARIMAX(filt_noise[d], order=ARMA_orders, enforce_invertibility=False,
                                    enforce_stationarity=False).fit(disp=False)
            model_noise_d = model.predict(2000, 3999, dynamic=True)
            model_noise_d = np.concatenate((filt_noise[d, 0:len(filt_noise[d])/2], model_noise_d), axis=0)


            if d == 0:
                model_noise = model_noise_d
                if self.dim == 1: model_noise = np.array([model_noise_d])
            else:
                model_noise = np.vstack((model_noise, model_noise_d))


        return model_noise
