import nengo
import numpy as np
import math
from nengo.utils.ensemble import tuning_curves
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, RegularGridInterpolator
from nengo.utils.numpy import meshgrid_nd
from scipy import signal
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import arma_generate_sample
import copy
import time

DIMENSIONS_KEY = 'dimensions'
RADIUS_KEY = 'radius'



class Mapping:
    """
        A class holding information about a mapping.

        Parameters
        ----------
        function: callable
            function defining the output of a mapping
        size_in: int
            number of input dimensions
        size_out: int
            number of output dimensions
        primary_dims: list of ints
            the dimensions used to compute the function
            i.e. f(x0, x1) = x0 has primary dimension [0]
        out_dims: list of ints
            dimensions computed by the function
            i.e. f(x0, x1) = [x0+x1, x0-x1, 0] has out_dims [0,1], but not 2
    """

    def __init__(self, function, size_in, size_out, primary_dims, out_dims):
        self.function = function
        self.size_in = size_in
        self.size_out = size_out
        self.primary_dims = primary_dims
        self.out_dims = out_dims


class BiasModel:
    """
        Given inputs, evaluate the bias at each output dimension. This
        involves projecting non primary dimensions into one dimension,
        and structuring data in the right format for interpolation models

        Parameters
        ----------
        mapping: Mapping object
            information about the function for which the bias is modeled
        functions_by_out_dims: list of callables
            bias functions for each output dimension defined the mapping
            i.e. if mapping.out_dims = [2,3], and functions_by_out_dims = [function1, function2],
            function1 computes dimension 2 and function2 computes dimension 3

    """

    def __init__(self, mapping, functions_by_out_dims):
        self.mapping = mapping
        self.functions_by_out_dims = functions_by_out_dims

    def eval(self, vals):
        ret = np.zeros((vals.shape[0], self.mapping.size_out))

        # if all input dimensions are used
        if len(self.mapping.primary_dims) == self.mapping.size_in:
            if len(self.mapping.primary_dims) == 1:
                for i, dim in enumerate(self.mapping.out_dims):
                    # gets ValueError: buffer source array is read-only if no deepcopy TODO: investigate why
                    ret[:,dim] = self.functions_by_out_dims[i](copy.deepcopy(vals[:,0]))
            elif len(self.mapping.primary_dims) == 2:
                for i, dim in enumerate(self.mapping.out_dims):
                    ret[:,dim] = self.compute_2d_bias_function(self.functions_by_out_dims[i], vals[:,0], vals[:,1])
            else:
                for i, dim in enumerate(self.mapping.out_dims):
                    ret[:,dim] = self.functions_by_out_dims[i](vals, method='linear')

        # if primary dims are a subset of input dimensions
        else:
            if len(self.mapping.primary_dims) == 1:
                # project input dimensions into 2D, one consisting of primary dim, the other
                # containing norm of the rest of the dimension
                primary_coords = vals[:,self.mapping.primary_dims[0]]
                projected_coords = np.array([np.linalg.norm(
                    np.concatenate((x[0:self.mapping.primary_dims[0]], x[self.mapping.primary_dims[0]+1:]))
                ) for x in vals])

                for i, dim in enumerate(self.mapping.out_dims):
                    ret[:,dim] = self.compute_2d_bias_function(self.functions_by_out_dims[i], primary_coords, projected_coords)
            elif len(self.mapping.primary_dims) == 2:
                # should be similar as above
                raise NotImplementedError
            else:
                raise NotImplementedError

        return ret

    def compute_2d_bias_function(self, function, dim1, dim2):
        if isinstance(function, RectBivariateSpline):
            return function(dim1, dim2, grid=False)
        elif isinstance(function, RegularGridInterpolator):
            return function(np.concatenate((dim1.reshape(-1,1), dim2.reshape(-1,1)), axis=1), method='linear')
        else:
            print type(function)
            raise NotImplementedError


class NoiseModel:
    """
        Given inputs, generate_noise() computes the noise value corresponding
        to each input step. It does so by finding the norm of each input,
        finding models built with noises of similar radii, and calculating
        the weighted sum of two noise models.

        Parameters
        ----------
        radii: list of floats or integers
            radii at which noises were sampled, sorted
        models: 2D array of ARMA objects
            The first dimension is ordered by the radii. Each index contains ARMA
            models fit to noises of the radius in that index, as contained in the
            radii parameter. The second dimension orders the ARMA model for noises
            of each output dimension of the ensemble

    """

    def __init__(self, radii, models, variances):
        self.noise_radii = radii
        self.noise_models_by_range = models
        self.noise_variances = variances

    def generate_noise(self, inp):
        out_shape = (len(inp), len(self.noise_models_by_range[0]))

        # for models built at each noise radius, we generate a noise
        # we do this first because generating noise is faster when done in batch
        noises_by_range = self._generate_noises_by_range(out_shape)

        # if we had built only one noise model, simply output the generated noise
        if len(self.noise_radii) == 1:
            return noises_by_range[0]
        else:
            noise = np.zeros(out_shape)
            inp_norm = np.linalg.norm(inp, axis=1)
            for step, val in enumerate(inp_norm):
                i = np.searchsorted(self.noise_radii, val)
                if i == 0:  # if norm is smaller than the smallest radius we anticipated
                    noise[step,:] = noises_by_range[0][step,:] # use the noise for smallest radius
                elif i == len(self.noise_radii): # if norm is greater than anticipated,
                    noise[step,:] = noises_by_range[-1][step,:] # use the noise for largest radius
                else:   # else, calculate the weighted sum using two most similar noise models
                    noise[step,:] = self._calculate_weighted_sum(
                        val,
                        noises_by_range[i-1][step,:],
                        noises_by_range[i][step,:],
                        i-1, i, out_shape[1]
                    )
            return noise

    def _generate_noises_by_range(self, out_shape):
        noises_by_range = []

        for i, model in enumerate(self.noise_models_by_range):
            noises_by_range.append(np.zeros(out_shape))
            for dim in range(out_shape[1]):
                noises_by_range[i][:,dim] = arma_generate_sample(
                    ar=np.r_[1, -model[dim].arparams],
                    ma=np.r_[1, model[dim].maparams],
                    nsample=out_shape[0],
                    sigma=model[dim].sigma2**0.5
                ).reshape(out_shape[0])

        return noises_by_range

    def _calculate_weighted_sum(self, val, noise_i, noise_j, i, j, dimensions):
        radii_gap = self.noise_radii[j] - self.noise_radii[i]
        w_i = (self.noise_radii[j] - val)/radii_gap
        w_j = (val - self.noise_radii[i])/radii_gap
        weighted_sum = w_i*noise_i + w_j*noise_j

        for dim in range(dimensions):
            std_i = self.noise_variances[i][dim]**0.5
            std_j = self.noise_variances[j][dim]**0.5
            scalar_factor = (w_i*std_i+w_j*std_j) * ( ((w_i*std_i)**2 + (w_j*std_j)**2) ** (-0.5) )
            weighted_sum[dim] *= scalar_factor * 1.2

        return weighted_sum




class SurrogateEnsemble(object):
    """
        Parameters
        ----------
        ens_config: dict
            contains the parameters that define an ensemble which we want to emulate
        connection: Connection object
            connection coming out of the ensemble which want to emulate
            NOTE: using this interface to leverage formatting done in Connection
        function_components: list of Mapping objects
            The function computed by the connection broken into linear combinations.
            Each component ideally has low primary_dims, so that bias model can be
            built efficiently for each.
            TODO: ideally we want a better interface as opposed to having user define
            this.
    """


    # ratio with respect to ensemble radius, which we expect to receive input
    INPUT_RANGE_RATIO = 3

    def __init__(self, ens_config, connection, function_components, dt=0.001):
        self.ens_config = ens_config
        self.connection = connection
        self.function_components = function_components
        self.dt = dt

        self.training_tranges = []
        self.training_ens_inputs = []
        self.training_ens_outputs = []
        self.training_sim_noises = []


        self.bias_models = []
        self.bias_models2 = []
        self.bias_models3 = []
        self.noise_radii = []
        self.noise_model = []


        self.trange = []

        self.ideal_output = []
        self.est_noise_values = []
        self.est_bias_values = []
        self.est_bias_values2 = []
        self.est_bias_values3 = []
        self.surrogate_output = []


        self.sim_bias_values = []
        self.sim_noise_values = []
        self.actual_output = []
        self.filtered_actual_output = []

        self.spike_freqs = [None]*connection.size_out
        self.spike_PSDs = [None]*connection.size_out
        self.model_freqs = [None]*connection.size_out
        self.model_PSDs = [None]*connection.size_out

        self.RMSE = []



    def build(self, noise_sampling_length=0.5, noise_sampling_steps=3, seed=None):
        """
            This method is called to build bias and noise models

            1. Build networks
            2. Sample points and build bias models

            3. Build networks for each input range
            4. Extract noise components and fit ARMA parameters


            Parameters
            -----------
            noise_sampling_length: float
                length of simulation to obtain noise sample, for each input range
            noise_sampling_steps: int
                number of input ranges to obtain noise samples
        """


        # Build bias

        ens_dimensions = self.ens_config.get(DIMENSIONS_KEY, 1)

        # If function is the identity
        if self.connection.function is None:
            sim, ens, out_conn, p_in, p_out = self.build_network(
                np.zeros(ens_dimensions), self.connection, None, np.array(1.0), self.dt, seed
            )
            # We break the identity function into components with low number of primary dimensions
            # i.e. f([x1, x2]) = [x1, x2] => f([x1, x2]) = [x1, 0] + [0, x2]
            for i in range(connection.size_in):
                mapping = Mapping(
                    lambda x: x, connection.size_in, connection.size_in, [i], [i]
                )
                bias_functions_by_out_dims = self.build_bias_functions(sim, ens, out_conn, mapping)
                self.bias_models.append(BiasModel(mapping, bias_functions))
        else:
            # build model for each components with low number of primary dimensions
            for mapping in self.function_components:
                sim, ens, out_conn, p_in, p_out = self.build_network(
                    np.zeros(ens_dimensions), self.connection, mapping.function, np.array(1.0), self.dt, seed
                )
                functions_by_out_dims  = self.build_bias_functions(sim, ens, out_conn, mapping, 'linear')
                self.bias_models.append(BiasModel(mapping, functions_by_out_dims))


        # Build noise

        # generate input ramps for each range input of radius
        noise_radii, ramp_inputs = self._generate_ramp_inputs(
            noise_sampling_length, noise_sampling_steps
        )
        noise_models = []
        variances = []
        for inp in ramp_inputs:
            sim, ens, out_conn, p_in, p_out = self.build_network(
                inp, self.connection, self.connection.function, self.connection.transform, self.dt, seed
            )
            sim.run(noise_sampling_length)
            # code for monitoring
            self.training_tranges.append(sim.trange())
            self.training_ens_inputs.append(sim.data[p_in])
            self.training_ens_outputs.append(sim.data[p_out])
            variance, model = self.build_noise_model(sim, ens, sim.data[p_in], sim.data[p_out], out_conn)
            variances.append(variance)
            noise_models.append(model)
        self.noise_model = NoiseModel(noise_radii, noise_models, variances)


    def _generate_ramp_inputs(self, noise_sampling_length, noise_sampling_steps):
        """
            Generate a list of callables that return inputs of radii ranges

            1. Determine the ranges of input radii
                i.e. if radius=1 and noise_sampling_steps=3, use [0, 0.5, 1]
            2. Construct ramp inputs for the input radii
        """


        ens_radius = self.ens_config.get(RADIUS_KEY, 1.0)
        ens_dimensions = self.ens_config.get(DIMENSIONS_KEY, 1)

        #
        input_radii = np.linspace(
            0, SurrogateEnsemble.INPUT_RANGE_RATIO*ens_radius,
            num=noise_sampling_steps
        )
        input_means = self._generate_hypersphere_points(ens_dimensions, input_radii)
        input_ramp_ranges = self._generate_hypersphere_points(
            ens_dimensions, np.full((noise_sampling_steps,), ens_radius*0.05)
        )

        start_points = np.array(input_means) - np.array(input_ramp_ranges)
        end_points = np.array(input_means) + np.array(input_ramp_ranges)

        def generate_ramp(noise_sampling_length, val_start, val_end):
            def ramp(t):
                return val_start + \
                    ( (val_end - val_start) * t/noise_sampling_length )
            return ramp

        ramps = []
        for val_start, val_end in zip(start_points, end_points):
            ramps.append(generate_ramp(noise_sampling_length, val_start, val_end))

        return input_radii, ramps


    def build_network(self, inp, connection, function, transform, dt, seed):
        """
            Build network given input, connection, function and transform
        """

        function = connection.function if function is None else function
        transform = connection.transform if transform is None else transform

        model = nengo.Network(seed=seed)
        with model:
            ensemble = nengo.Ensemble(**self.ens_config)
            in_node = nengo.Node(inp, size_out=ensemble.dimensions)
            in_conn = nengo.Connection(in_node, ensemble, synapse=None)

            out_node = nengo.Node(size_in=connection.size_out)

            out_conn = nengo.Connection(ensemble, out_node,
                                        synapse=connection.synapse,
                                        function=function,
                                        transform=transform,
                                        solver=connection.solver)

            p_in = nengo.Probe(in_node, synapse=connection.synapse)
            p_out = nengo.Probe(out_node, synapse=None)

        sim = nengo.Simulator(model, dt)

        return sim, ensemble, out_conn, p_in, p_out


    def build_bias_functions(self, sim, ens, out_conn, mapping, interporlation='linear'):
        """
            Build bias functions given a built network

            1. generate sample points
            2. calculate bias at those points
            3. interpolate on the points
        """


        eval_points, grid_points = self._generate_eval_points(ens.radius, mapping)
        sampled_bias = self._calc_bias(sim, ens, eval_points, out_conn, mapping.function)

        self.bias_eval_points = eval_points
        self.sampled_bias = sampled_bias

        functions_by_out_dims = []

        if len(grid_points) == 1:
            for i in mapping.out_dims:
                functions_by_out_dims.append(
                    interp1d(grid_points[0].ravel(), sampled_bias[:,i].reshape(-1), kind=interporlation)
                )
        elif len(grid_points) == 2:
            x, y = grid_points
            for i in mapping.out_dims:
                functions_by_out_dims.append(
                    RectBivariateSpline(x, y, sampled_bias[:,i].reshape(x.size, y.size))
                )

                # functions_by_out_dims.append(
                #     RegularGridInterpolator((x, y), sampled_bias[:,i].reshape(x.size, y.size), method='linear')
                # )
        else:
            shapes = map(lambda x: len(x), grid_points)
            for i in mapping.out_dims:
                functions_by_out_dims.append(
                    RegularGridInterpolator(grid_points, sampled_bias[:,i].reshape(shapes), method='linear')
                )

        return functions_by_out_dims


    def _generate_eval_points(self, radius, mapping):
        """
            Given the radius of the ensemble and a mapping,
            generate points to sample bias

            Roughly:
            1. Uniformly generate points from each primary dimension
            2. Uniformaly generate points from the to-be-projected dimension
            3. Generate multi-dimensional points corresponding to the secondary dimensions with
                the radii from step 2

            Returns
            --------
            eval_points: ndarray of shape (# of sample points, # of input dimensions)
                the secondary dimensions are contained separately
            grid_points_by_dims: the coordinates of each dimension, in increasing order.
                Secondary dimensions are projected into one

        """

        NUM_POINTS_PER_DIM = 100 # TODO move this somewhere else

        num_secondary_dims = mapping.size_in - len(mapping.primary_dims)
        num_grid_dims = len(mapping.primary_dims) + (num_secondary_dims > 0)

        ratio = SurrogateEnsemble.INPUT_RANGE_RATIO / (num_grid_dims**0.5)
        sample_range = (-ratio*radius, ratio*radius)

        grid_points_by_dims = []
        for i in mapping.primary_dims:
            grid_points_by_dims.append(np.linspace(*sample_range, num=NUM_POINTS_PER_DIM))

        num_secondary_dims = mapping.size_in - len(mapping.primary_dims)

        if num_secondary_dims == 0:
            grid_meshes = meshgrid_nd(*grid_points_by_dims)
            if len(grid_meshes) == 1:
                eval_points = grid_meshes[0].reshape(-1,1)
            else:
                eval_points = np.concatenate((map(lambda x: x.reshape(-1,1), grid_meshes)), axis=1)
        else:
            secondary_dims_max_norm = ratio*radius

            secondary_dim_units = np.linspace(0, secondary_dims_max_norm, NUM_POINTS_PER_DIM)
            grid_points_by_dims.append(secondary_dim_units)
            sphere_points = self._generate_hypersphere_points(num_secondary_dims, secondary_dim_units)

            # build a mapping from the projected point to the original coordinates
            secondary_unit_to_points  = dict(zip(secondary_dim_units, sphere_points))
            # build meshes
            grid_meshes = meshgrid_nd(*grid_points_by_dims)

            # recover full coordinates of the secondary dimensions from the projected points
            eval_points = np.array([secondary_unit_to_points[x] for x in grid_meshes[-1].reshape(-1)])
            # insert the primary coordinates
            for i, dim in enumerate(mapping.primary_dims):
                eval_points = np.insert(eval_points, [dim], grid_meshes[i].reshape(-1,1), axis=1)


        return eval_points, grid_points_by_dims


    def _generate_hypersphere_points(self, dims, radii):
        """
            Uniformly generate points of the given # of dimensions and the
            given radii

        """
        points = nengo.dists.UniformHypersphere(surface=True).sample(
            len(radii), dims)

        radii_points = [radius*point for radius, point in zip(radii, points)]

        return radii_points

    def _calc_bias(self, sim, ens, eval_points, out_conn, function):
        """
            Extract the bias component given evaluation points by taking
            the difference between the decoded and ideal output
        """

        _, encoded_rates = tuning_curves(ens, sim, inputs=eval_points)
        static_output = np.dot(encoded_rates, sim.data[out_conn].weights.T)
        out_shape = static_output.shape

        ideal_output = np.apply_along_axis(function, 1, eval_points).reshape(out_shape) \
            if function is not None else eval_points
        bias =  static_output - ideal_output

        return bias


    def build_noise_model(self, sim, ens, ens_input, ens_output, out_conn):
        """
            Given a simulation, build noise model

            1. Extract noise
            2. For each output dimension, build ARMA model
        """

        sim_noise = self._calc_noise(sim, ens, ens_input, ens_output, out_conn)
        self.training_sim_noises.append(sim_noise)  # CODE FOR MONITORING

        variance_by_dimensions = []
        models_by_dimensions = []
        BUFFER_STEPS = 50  # this skips the sometimes abnormal noise at the beginning
        for dim in range(sim_noise.shape[1]):
            variance, model = self._build_elementary_noise_model(sim_noise[BUFFER_STEPS:, dim], (2,2))
            variance_by_dimensions.append(variance)
            models_by_dimensions.append(model)

        return variance_by_dimensions, models_by_dimensions

    def _build_elementary_noise_model(self, noise, order):
        """
            Given single dimensional noise values, fit ARMA
            parameters to them
        """

        MIN_SAMPLE_NOISE_LENGTH = 200 # TODO: move this somewhere else
        TRIAL_MAX = 3

        if len(noise) < MIN_SAMPLE_NOISE_LENGTH:
            raise Exception("noise sample too short")

        count = 1
        seed = 12345 # TODO clean up
        while count <= TRIAL_MAX:
            try:
                np.random.seed(seed)
                # need to set default params to zeros or optimizer does not converge TODO: investigate why
                model = ARMA(noise, order=order).fit([0,0,0,0], trend='nc', disp=False)
                break
            except Exception as e:  #sometimes the solver throws errors because solution does not converge
                print e
                count += 1
                seed += 1   # TODO: it seems that changing seed does not help solver to converge -> try regenerating the noise

        return np.var(noise), model


    def _calc_noise(self, sim, ens, eval_points, actual_output, out_conn):
        """
            Extract the noise component by taking the difference between the actual
            output and the statically decoded output
        """

        assert len(eval_points) == len(actual_output)

        _, encoded_rates = tuning_curves(ens, sim, inputs=eval_points)
        static_output = np.dot(encoded_rates, sim.data[out_conn].weights.T)
        noise = actual_output - static_output

        return noise


    def emulate(self, inp, out_shape, seed=6):
        """
            Given inputs, calculate the ideal output, estimate the bias,
            and generate noise. Sum them to emulate a real simulation
        """


        self.ideal_output = np.apply_along_axis(self.connection.function, 1, inp).reshape(out_shape)

        import time
        pre_transform_est_bias_values = sum(
            model.eval(inp).reshape(out_shape) for model in self.bias_models
        )
        self.est_bias_values = np.dot(pre_transform_est_bias_values, self.connection.transform.T)

        self.est_noise_values = self.noise_model.generate_noise(inp)

        self.surrogate_output = self.ideal_output + self.est_bias_values + self.est_noise_values

        return self.surrogate_output


    def test_performance(self, inp, length=6, seed=None):
        """
            Runs simulation for a period, and compares emulated output and actual output
        """
        sim, ens, out_conn, p_in, p_out = self.build_network(
            inp, self.connection, self.connection.function, self.connection.transform, self.dt, seed
        )
        sim.run(length)


        self.trange = sim.trange()
        self.input = sim.data[p_in]
        self.actual_output = sim.data[p_out]
        out_shape = self.actual_output.shape


        self.emulate(sim.data[p_in], out_shape)


        _, sim_rates = tuning_curves(ens, sim, inputs=sim.data[p_in])
        self.sim_static_output = np.dot(sim_rates, sim.data[out_conn].weights.T)

        self.sim_bias_values = self._calc_bias(sim, ens, sim.data[p_in], out_conn, self.connection.function)
        self.sim_noise_values = self._calc_noise(sim, ens, sim.data[p_in], sim.data[p_out], out_conn)

        # power spectral density analysis
        for i in range(out_shape[1]):
            self.spike_freqs[i], self.spike_PSDs[i] = signal.periodogram(self.sim_noise_values[:,i].ravel(), fs=length/self.dt, window='hanning')
            self.model_freqs[i], self.model_PSDs[i] = signal.periodogram(self.est_noise_values[:,i].ravel(),fs=length/self.dt,window='hanning')

