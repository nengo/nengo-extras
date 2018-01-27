import nengo
import numpy as np

class SurrogateNode(nengo.Node):
    def __init__(self, sim, ensemble, mode):
        # go do all your build stuff

        self.sim = sim
        self.ensemble = ensemble
        self.mode = mode
        super(SurrogateNode, self).__init__(sim.dt, size_in=ensemble.dimensions, size_out=ensemble.dimensions)

    def step(self, t, x):
        # compute one time step output for a value of x
        t_index = np.where(self.time == t)

        bias = self.createBiasModel(t_index, self.mode)
        noise = self.createNoiseModel(t_index, self.ARMA_orders)
        return x + bias + noise


def replace_with_surr_node(self, model, ens, sim):
    model.ensembles.remove(ens)

    with model:
        surrogate = SurrogateNode(sim, ens)

    for c in model.connections[:]:
        if c.post_obj is ens:
            model.connections.remove(c)
            with model:
                nengo.Connection(c.pre, surrogate[c.post_slice],
                                 transform=c.transform,
                                 synapse=c.synapse)
    for c in model.connections[:]:
        if c.pre_obj is ens:
            model.connections.remove(c)
            with model:
                nengo.Connection(surrogate[c.pre_slice], c.post,
                                 transform=c.transform,
                                 synapse=c.synapse)

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
        ind = 
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
