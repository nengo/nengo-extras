import nengo
import numpy as np
import math
from nengo.utils.ensemble import tuning_curves
from scipy.interpolate import interp1d
from scipy import signal
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import arma_generate_sample

class SurrogateEnsemble(object):
    def __init__(self, ensemble_config, function, synapse=0.005, dt=0.001):
        self.ens_config = ensemble_config
        self.function = function
        self.synapse = synapse
        self.dt = dt

        self.ens_input = []
        self.ens_output = []

        # Params for bias model
        self.bias_model = None
        self.noise_model = None


        self.trange = []
        self.ideal_output = []
        self.actual_output = []
        self.filtered_actual_output = []
        self.surrogate_output = []

        self.sim_bias_values = []
        self.est_bias_values = []

        # Params for noise model
        self.sim_noise_values = []
        self.filt_noise_values = []
        self.est_noise_values = []

        self.spike_freq = []
        self.spike_PSD = []
        self.filt_freq = []
        self.filt_PSD = []
        self.model_freq = []
        self.model_PSD = []

        self.RMSE = []
        self.surr_output = []


    def build(self, length=0.5, seed=None):
        sim, ens, out_conn, p_in, p_out, p_ensemble = self.run_simulation(
            nengo.processes.WhiteSignal(period=10, high=1, rms=0.2, seed=seed), length, seed)

        self.training_trange = sim.trange()
        self.ens_input = sim.data[p_in]
        self.ens_output = sim.data[p_out]
        self.ens_probed = sim.data[p_ensemble]
        self.training_ideal_output = np.apply_along_axis(self.function, 0, sim.data[p_in])

        self.bias_model = self.build_bias_model(sim, ens, out_conn)
        self.noise_model = self.build_noise_model(sim, ens, out_conn)


    def run_simulation(self, input, length, seed):
        model = nengo.Network(seed=seed)
        with model:
            ensemble = nengo.Ensemble(**self.ens_config)
            in_node = nengo.Node(input)
            in_conn = nengo.Connection(in_node, ensemble, synapse=self.synapse)

            out_node = nengo.Node(size_in=self.ens_config.get('dimensions',1))
            out_conn = nengo.Connection(ensemble, out_node,
                                      function=self.function,
                                      synapse=0.0)

            p_in = nengo.Probe(in_node, synapse=self.synapse)
            p_out = nengo.Probe(out_node, synapse=0.0)

            # p_ensemble = nengo.Probe(ensemble, synapse=0.0)


        with nengo.Simulator(model, self.dt) as sim:
            sim.run(length)

        return sim, ensemble, out_conn, p_in, p_out, p_ensemble


    def build_bias_model(self, sim, ens, out_conn):
        sample_range = (-3*ens.radius, 3*ens.radius)
        # eval_points = nengo.dists.Uniform(*sample_range).sample(30, ens.dimensions)
        eval_points = np.linspace(*sample_range, num=400).reshape((400,1))
        sampled_bias = self.calc_bias(sim, ens, out_conn, eval_points)

        if ens.dimensions == 1:
            bias_model = interp1d(eval_points.ravel(), sampled_bias,
                                            axis=0,
                                            kind='linear',
                                            fill_value=np.array([0]))
        else:
            pass
            # supa hard


        return bias_model


    def calc_bias(self, sim, ens, out_conn, eval_points):
        _, sim_rates = tuning_curves(ens, sim, inputs=eval_points)
        sim_static_output = np.dot(sim_rates, sim.data[out_conn].weights.T)
        ideal_output = np.array([self.function(x) for x in eval_points])
        sim_bias = sim_static_output - ideal_output

        return sim_bias


    def build_noise_model(self, sim, ens, out_conn):
        sim_noise = self.calc_noise(sim, ens, out_conn, self.ens_input, self.ens_output)
        filt_sim_noise = self.lowpass_filter(sim_noise, sim.trange())

        # sim_noise_probed = self.calc_noise(sim, ens, out_conn, self.ens_input, self.ens_probed)
        # self.training_filt_sim_noise_probed = self.lowpass_filter(sim_noise_probed, sim.trange())

        _, sim_rates = tuning_curves(ens, sim, inputs=self.ens_input)
        self.training_sim_static_output = np.dot(sim_rates, sim.data[out_conn].weights.T)

        self.training_sim_noise = sim_noise
        self.training_filt_sim_noise = filt_sim_noise

        np.random.seed(12345)
        noise_model = ARMA(filt_sim_noise, order=(3,3)).fit([0,0,0,0,0,0], trend='nc', disp=False)

        return noise_model

    def calc_noise(self, sim, ens, out_conn, eval_points, actual_output):
        assert len(eval_points) == len(actual_output)

        _, sim_rates = tuning_curves(ens, sim, inputs=eval_points)
        sim_static_output = np.dot(sim_rates, sim.data[out_conn].weights.T)
        sim_noise = actual_output - sim_static_output

        return sim_noise


    def lowpass_filter(self, noise, trange):
        # TODO multi dimension
        sys_PSC = signal.TransferFunction(1, [self.synapse, 1])
        filt_noise = signal.lsim(sys_PSC, noise, trange)[1]

        return filt_noise


    def test_performance(self, input, length=6, seed=None):
        sim, ens, out_conn, p_in, p_out, p_ensemble = self.run_simulation(input, length, seed)

        self.trange = sim.trange()
        self.actual_output = sim.data[p_out]
        self.filtered_actual_output = self.lowpass_filter(self.actual_output, sim.trange())

        self.ideal_output = np.apply_along_axis(self.function, 0, sim.data[p_in])
        out_shape = self.ideal_output.shape
        self.est_bias_values = self.bias_model(sim.data[p_in]).reshape(out_shape)
        self.est_noise_values = arma_generate_sample(
            ar=np.r_[1, -self.noise_model.arparams],
            ma=np.r_[1, self.noise_model.maparams],
            nsample=len(self.actual_output),
            sigma=self.noise_model.sigma2**0.5
        ).reshape(out_shape)


        self.surrogate_output = self.ideal_output + self.est_bias_values + self.est_noise_values

        self.sim_bias_values = self.calc_bias(sim, ens, out_conn, sim.data[p_in])
        self.sim_noise_values = self.calc_noise(sim, ens, out_conn, sim.data[p_in], sim.data[p_out])
        self.filt_noise_values = self.lowpass_filter(self.sim_noise_values, sim.trange())


        self.spike_freq, self.spike_PSD = signal.periodogram(self.sim_noise_values.ravel(), fs=length/self.dt, window='hanning')
        self.filt_freq, self.filt_PSD = signal.periodogram(self.filt_noise_values.ravel(),fs=length/self.dt,window='hanning')
        self.model_freq, self.model_PSD = signal.periodogram(self.est_noise_values.ravel(),fs=length/self.dt,window='hanning')


