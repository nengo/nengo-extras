# Surrogate API that does not require full spiking simulation
import random
import nengo
import nengo_normal_form

import nengo.utils.numpy as npext
import numpy as np
import pandas as pd
import time
import SurrogateNode

from nengo.utils.ensemble import tuning_curves
from numpy.polynomial.polynomial import polyfromroots
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from scipy import signal
from statsmodels.tsa.statespace.sarimax import SARIMAX

###################
# Surrogate Model #
###################

class SurrogateModel(object):    
    def __init__(self, model, ensemble, dim, sim_time, dt, input, 
                    ARMA_orders=[2,0,2], tau_syn=0.05, seed=None):
        
        self.dim = dim                           # model dimension
        self.stim_probe = None
        self.pop_probe = None
        self.input = input
        self.sim_time = sim_time                 # simulated time
        self.dt = dt                             # time step
        self.trange = np.linspace(0, sim_time, num=int(sim_time/dt))
        self.ARMA_orders = ARMA_orders
        self.tau_syn = tau_syn
        self.ensemble = ensemble                 # desired ensemble
        self.model = model                       # model where the desired ensemble belongs

        self.eval_points = []
        self.rates = []
        self.decoders = []
        self.static_output = []

        # Params for bias model
        self.bias_values_sim = []                # time-independent difference between input and static_output
        self.bias_values_est = []                # estimate of simulated bias

        # Params for noise model
        self.noise_values_est = []               # estimate of noise (difference between output and static_output),
                                                 # using random gaussian distributed samples

        self.output = []                         # surrogate output
        self.simulated_time = []                 # sim tic_toc
        self.build()


    #############################
    # SURROGATE MODEL FUNCITONS #
    #############################

    def build(self):
        """
        Build the surrogate model; 
        FOR NOW: assume you are only interested in the last population
        TODO: Make it applicable to any ensemble

        Steps:
        1. Calculate the rates of the ensemble based on the input
        2. Find the decoders of the ensembles
        3. Calculate the static output of the ensemble
        """
        pop = self.ensemble
        n_neurons = pop.n_neurons

        with nengo.Simulator(self.model) as sim:
            eval_points, rates = tuning_curves(pop, sim, inputs=self.input.T)

        # Find decoders
        conn_dec = None
        is_decoded = False
        
        # Fetch all the decoded connections of the desired ensemble
        for conn in self.model.connections:
            if self.ensemble == conn.pre:
                if type(conn.post) == nengo.node.Node:
                    is_decoded = True
                    conn_dec = conn
        
        # Create a node for decoder if it doesn't exist
        model = self.model
        if not is_decoded:        
            with model:
                for conn in model.connections:
                    if pop == conn.pre:
                        node = nengo.Node(None, size_in=self.dim, label='node_decode') # For decoders
                        conn_dec = nengo.Connection(pop, node)
                        break

            is_decoded = True
            self.model = model
            with nengo.Simulator(self.model) as sim:
                eval_points, rates = tuning_curves(pop, sim, inputs=self.input.T)
                pass

        self.eval_points = eval_points
        self.rates = rates.T
        self.decoders = sim.data[conn_dec].weights
        self.static_output = np.dot(self.decoders, self.rates)

        # Replace the ensemble with surrogate node
        SurrogateNode.replace_with_surr_node(model, ens, sim, static_output)


    def run(self):
        """
        Calculates the surrogate model based on the estimated noise and bias terms;
        SurrogateModel.run() should have ran in order to run this function

        NOTE: surr_output = input + bias_est + noise_est

        Steps:
        1. Ensure that the model has been simulated
        2. Estimate the bias and noise term
        3. Calculate the surr_model output
        """
        # tic = time.clock()

        try:
            assert len(self.static_output) != 0
        except:
            raise ValueError("Surrogate model has not been simulated")
        else:
            with nengo.Simulator(self.sim) as sim:
                sim.run(self.sim_time)
                        
            self.output = surr_output

            # TODO: Consider multidimensional case

        # toc = time.clock()

        # self.simulated_time = toc - tic
        # print "Simulating finished in {}".format(self.simulated_time)

    ########################
    # BIAS MODEL FUNCTIONS #
    ########################

    def createBiasModel(self, t, mode="lin_interp"):
        """
        Creates bias model of the surrogate model.
        mode: different mode of bias approximation (types: lin_interp, poly_reg, fourier_reg)
        
        Steps:
        1. Calculate the simulated bias
        2. Estimate the bias term based on the simulated bias
        """
        if self.dim == 1:
            self.calcSimBias(t)
            self.estimateBias(t, mode)
        
        # TODO: Finish implementing for multidimensional case
    
    def calcSimBias(self, t):
        """
        Obtains samples of bias (distortion) error which can then later be fit to a 
        model. Returns an array of bias errors at each eval points for each 
        origin and an ideal values (ideal). 

        Note: actual = ideal + bias
        """
        # ind = 
        bias = self.static_output - self.input
        self.bias_values_sim = bias

    def estimateBias(self, t, mode):
        """
        Estimates the simulated bias by interpolation method specified by "mode"

        Steps:
        1. Calculate the interpolation function using the exiting eval_points and the simulated bias term
        2. Generate new points based on the eval_points
        3. Calculate the estimated bias using the interpolation function from Step 1
        
        Params:
        mode: interpolation method
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
                                abs(eval_points[-1])), dist="gaussian")
                new_points = np.sort(new_points)
                self.bias_values_est = func(new_points).squeeze()
            
            # elif mode == "poly_reg":
                # Interpolate using poly_reg

            # elif mode == "fourier":
                # Interpolate using fourier
                
        # TODO: Complete method for bias interpolation for multidimensional case

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

    def createNoiseModel(self, ARMA_orders):
        """
        Create noise model for estimating the noise term of the surrogate model
        """
        model_noise = self.estimateNoise(ARMA_orders)
        self.noise_values_est = model_noise

    def estimateNoise(self, ARMA_orders):
        """
        Estimates the simulated noise using randomly distributed points

        Steps:
        1. Generate two random noise (gaussian distributed)
        2. Pass them through LPF filter for modelling the actual noise
        3. Generate a model noise by filtering the random noise with ARMA model

        Params:
        ARMA_orders: ARIMA model param; default is [2,0,2]
        """
        rand_noise_1 = np.random.randn(self.static_output.shape[0], self.static_output.shape[1])
        rand_noise_1 = self.filtNoise(rand_noise_1, 0.005)
        rand_noise_2 = np.random.randn(self.static_output.shape[0], self.static_output.shape[1])
        rand_noise_2 = self.filtNoise(rand_noise_2, 0.005)

        model_noise = self.fitARMAModel(rand_noise_1, rand_noise_2, ARMA_orders)

        return model_noise

    def filtNoise(self, noise, tau_syn):
        """
        Perform LPF on the noise for filtering

        Params:
        1. Noise: desired noise for filtering
        2. tau_syn: time constant at synapse; used for generating LPF
        """
        sys_PSC = signal.TransferFunction(1, [tau_syn, 1])
        for d in range(self.dim):
            filt_noise_d = signal.lsim(sys_PSC, noise[d], self.trange)[1] # interested only in yout
            if d == 0:
                filt_noise = filt_noise_d
            else:
                filt_noise = np.vstack((filt_noise, filt_noise_d))
        
        return filt_noise.reshape((self.dim,len(filt_noise)))

    def fitARMAModel(self, noise, rand_noise, ARMA_orders):
        """
        Fit ARMA model to spike noise spectrum; 
        returns the estimated noise based on rand_noise

        Params:
        noise: ARMA model fitting noise
        rand_noise: random noise that would be used to calculate 
                    the model noise based on the ARMA fit
        ARMA_orders: ARMA model params; default is [2,0,2]
        """
        orders = ARMA_orders 

        # Find ARMA model in each dimension
        for d in range(self.dim):
            model_d = SARIMAX(noise[d], order=orders, enforce_stationarity=False, 
                                enforce_invertibility=False)
            model_fit_d = model_d.fit(disp=-1)
            model_noise_d = model_fit_d.predict()

            # arma filtering
            model_d_rand = SARIMAX(rand_noise[d], order=orders, enforce_stationarity=False, 
                                enforce_invertibility=False)
            model_d_rand = model_d_rand.filter(model_fit_d.params)
            model_noise_d = model_d_rand.predict()
            
            # Formatting model_noise
            if d == 0:
                model_noise = model_noise_d
                if self.dim == 1: model_noise = np.array([model_noise_d])
            else:
                model_noise = np.vstack((model_noise, model_noise_d))

        return model_noise

