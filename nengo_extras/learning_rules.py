import warnings

import numpy as np

from nengo.builder import Builder, Signal
from nengo.builder.learning_rules import get_pre_ens, get_post_ens
from nengo.builder.operator import (
    DotInc, ElementwiseInc, Operator, Reset, SimPyFunc)
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleType
from nengo.params import (
    BoolParam, Default, EnumParam, FunctionParam, NumberParam)
from nengo.synapses import Lowpass


class DeltaRuleFunctionParam(FunctionParam):
    function_test_size = 8  # arbitrary size to test function

    def function_args(self, instance, function):
        return (np.zeros(self.function_test_size),)

    def coerce(self, instance, function):
        function_info = super(DeltaRuleFunctionParam, self).coerce(
            instance, function)

        function, size = function_info
        if function is not None and size != self.function_test_size:
            raise ValidationError(
                "Function '%s' input and output sizes must be equal" %
                function, attr=self.name, obj=instance)

        return function_info


class DeltaRule(LearningRuleType):
    r"""Implementation of the Delta rule.

    By default, this implementation pretends the neurons are linear, and thus
    does not require the derivative of the postsynaptic neuron activation
    function. The derivative function, or a surrogate function, for the
    postsynaptic neurons can be provided in ``post_fn``.

    The update is given by

    .. math::

        \delta W_ij = \eta a_j e_i f(u_i)

    where :math:`e_i` is the input error in the postsynaptic neuron space,
    :math:``a_j` is the jth presynaptic neuron (output) activity,
    :math:`u_i` is the ith postsynaptic neuron input,
    and :math:`f` is a provided function.

    Parameters
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    pre_tau : float
        Filter constant on the presynaptic output :math:`a_j`.
    post_fn : callable
        Function :math:`f` to apply to the postsynaptic inputs :math:`u_i`.
        The default of ``None`` means the :math:`f(u_i)` term is omitted.
    post_tau : float
        Filter constant on the postsynaptic input :math:`u_i`. This defaults to
        ``None`` because these should typically be filtered by the connection.
    post_target : string
        Which side of the learned connection to use for postsynaptic inputs.
        Can be ``"in"`` (the default) for input or ``"out"`` for output.
    """
    modifies = 'weights'
    probeable = ('delta', 'in', 'error', 'correction', 'pre', 'post')

    learning_rate = NumberParam(
        'learning_rate', low=0, readonly=True, default=1e-4)
    pre_tau = NumberParam('pre_tau', low=0, low_open=True, default=0.005)
    post_fn = DeltaRuleFunctionParam('post_fn', optional=True, default=None)
    post_tau = NumberParam(
        'post_tau', low=0, low_open=True, optional=True, default=None)
    post_target = EnumParam('post_target', values=('in', 'out'), default="in")

    def __init__(self, learning_rate=Default, pre_tau=Default,
                 post_fn=Default, post_tau=Default, post_target=Default):
        super(DeltaRule, self).__init__(learning_rate, size_in='post')
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn("This learning rate is very high, and can result "
                          "in floating point errors from too much current.")
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.post_fn = post_fn
        self.post_target = post_target

    @property
    def _argreprs(self):
        args = []
        if self.learning_rate != 1e-4:
            args.append("learning_rate=%g" % self.learning_rate)
        if self.pre_tau != 0.005:
            args.append("pre_tau=%f" % self.pre_tau)
        if self.post_fn is not None:
            args.append("post_fn=%s" % self.post_fn.function)
        if self.post_tau is not None:
            args.append("post_tau=%f" % self.post_tau)
        if self.post_target != 'in':
            args.append("post_target=%s" % self.post_target)

        return args


@Builder.register(DeltaRule)
def build_delta_rule(model, delta_rule, rule):
    conn = rule.connection

    # Create input error signal
    error = Signal(np.zeros(rule.size_in), name="DeltaRule:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error  # error connection will attach here

    # Multiply by post_fn output if necessary
    post_fn = delta_rule.post_fn.function
    post_tau = delta_rule.post_tau
    post_target = delta_rule.post_target
    if post_fn is not None:
        post_sig = model.sig[conn.post_obj][post_target]
        post_synapse = Lowpass(post_tau) if post_tau is not None else None
        post_input = (post_sig if post_synapse is None else
                      model.build(post_synapse, post_sig))

        post = Signal(np.zeros(post_input.shape), name="DeltaRule:post")
        model.add_op(SimPyFunc(post, post_fn, t=None, x=post_input,
                               tag="DeltaRule:post_fn"))
        model.sig[rule]['post'] = post

        error0 = error
        error = Signal(np.zeros(rule.size_in), name="DeltaRule:post_error")
        model.add_op(Reset(error))
        model.add_op(ElementwiseInc(error0, post, error))

    # Compute: correction = -learning_rate * dt * error
    correction = Signal(np.zeros(error.shape), name="DeltaRule:correction")
    model.add_op(Reset(correction))
    lr_sig = Signal(-delta_rule.learning_rate * model.dt,
                    name="DeltaRule:learning_rate")
    model.add_op(DotInc(lr_sig, error, correction, tag="DeltaRule:correct"))

    # delta_ij = correction_i * pre_j
    pre_synapse = Lowpass(delta_rule.pre_tau)
    pre = model.build(pre_synapse, model.sig[conn.pre_obj]['out'])

    model.add_op(Reset(model.sig[rule]['delta']))
    model.add_op(ElementwiseInc(
        correction.column(), pre.row(), model.sig[rule]['delta'],
        tag="DeltaRule:Inc Delta"))

    # expose these for probes
    model.sig[rule]['error'] = error
    model.sig[rule]['correction'] = correction
    model.sig[rule]['pre'] = pre


class STDP(LearningRuleType):
    r"""Spike-timing dependent plasticity rule.

    This is the traditional doublet STDP rule. It implements the equation

    .. math::

       \Delta \omega_{ij} = \sum_{si} \sum_{sj} W(t_j^{sj} - t_i^{si})

    where

    * :math:`i` indexes the presynaptic neuron
    * :math:`j` indexes the postsynaptic neuron
    * :math:`\omega_{ij}` is the weight on the connection from neuron
      :math:`i` to neuron :math:`j`
    * :math:`si` is the set of spikes produced by the presynaptic neuron
    * :math:`sj` is the set of spikes produced by the postsynaptic neuron
    * :math:`W(\cdot)` is a function denoting the desired weight change

    :math:`W` is typically computed as

    .. math::

       W(x) = A_+ \exp(-x/\tau_+) \quad &\text{ for } x>0 \\
       W(x) = -A_- \exp(x/\tau_-) \quad &\text{ for } x<0

    where

    * :math:`A_+` and :math:`A_-` are amplitude parameters
    * :math:`\tau_+` and :math:`\tau_-` are exponential decay time constant
      parameters (usually on the order of 10 ms)

    The function :math:`W` and parameter values are derived from
    experimental results, like those in Bi & Poo, 1998.

    See :doc:`examples/stdp` for more details.

    Parameters
    ----------
    pre_tau : float
       Time constant on the presynaptic trace decay.
    pre_amp : float
       Presynaptic trace amplitude.
    post_tau : float
       Time constant on the postsynaptic trace decay.
    post_amp : float
       Postsynaptic trace amplitude.
    bounds : string
       How to bound connection weights. Can be ``"hard"``, the default, which
       clips weights to be within ``min_weight`` and ``max_weight``; ``"soft"``,
       which decays large amplitude weights toward the min and max on each
       timestep; or ``"none"``, which allows weights to grow unbounded.
    max_weight : float
       Maximum connection weight. Only used if ``bounds`` is "soft" or "hard".
    min_weight : float
       Minimum connection weight. Only used if ``bounds`` is "soft" or "hard".
    """

    modifies = 'weights'
    probeable = ['pre_trace', 'post_trace', 'pre_scale', 'post_scale']

    learning_rate = NumberParam(
        'learning_rate', low=0, readonly=True, default=1e-9)
    pre_tau = NumberParam("pre_tau", low=0, low_open=True, default=0.0168)
    pre_amp = NumberParam("pre_amp", low=0, low_open=True, default=1.)
    post_tau = NumberParam("post_tau", low=0, low_open=True, default=0.0337)
    post_amp = NumberParam("post_amp", low=0, low_open=True, default=1.)
    bounds = EnumParam("bounds", values=("hard", "soft", "none"), default="hard")
    min_weight = NumberParam("min_weight", default=-0.3)
    max_weight = NumberParam("max_weight", default=0.3)

    def __init__(self, learning_rate=Default, pre_tau=Default, pre_amp=Default,
                 post_tau=Default, post_amp=Default, bounds=Default,
                 min_weight=Default, max_weight=Default):
        super(STDP, self).__init__(learning_rate, size_in=0)
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.bounds = bounds
        self.max_weight = max_weight
        self.min_weight = min_weight


class SimSTDP(Operator):
    def __init__(self, pre_activities, post_activities, pre_trace, post_trace,
                 pre_scale, post_scale, weights, delta, learning_rate,
                 pre_tau, post_tau, pre_amp, post_amp, bounds,
                 max_weight, min_weight):
        self.pre_activities = pre_activities
        self.post_activities = post_activities
        self.pre_trace = pre_trace
        self.post_trace = post_trace
        self.pre_scale = pre_scale
        self.post_scale = post_scale
        self.weights = weights
        self.delta = delta
        self.learning_rate = learning_rate
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.bounds = bounds
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities, weights]
        self.updates = [delta, pre_trace, post_trace, pre_scale, post_scale]

    def make_step(self, signals, dt, rng):
        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace = signals[self.pre_trace]
        post_trace = signals[self.post_trace]
        pre_scale = signals[self.pre_scale]
        post_scale = signals[self.post_scale]
        weights = signals[self.weights]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        # Could be configurable
        pre_ampscale = 1.
        post_ampscale = 1.

        if self.bounds == 'hard':

            def update_scales():
                pre_scale[...] = ((self.max_weight - weights) > 0.
                                  ).astype(np.float64) * pre_ampscale
                post_scale[...] = -((self.min_weight + weights) < 0.
                                    ).astype(np.float64) * post_ampscale
        elif self.bounds == 'soft':

            def update_scales():
                pre_scale[...] = (self.max_weight - weights) * pre_ampscale
                post_scale[...] = (self.min_weight + weights) * post_ampscale

        elif self.bounds == 'none':

            def update_scales():
                pre_scale[...] = pre_ampscale
                post_scale[...] = -post_ampscale

        def step_stdp():
            update_scales()
            pre_trace[...] += ((dt / self.pre_tau)
                               * (-pre_trace
                                  + self.pre_amp * pre_activities))
            post_trace[...] += ((dt / self.post_tau)
                                * (-post_trace
                                   + self.post_amp * post_activities))
            delta[...] = (alpha
                          * (pre_scale
                             * pre_trace[np.newaxis, :]
                             * post_activities[:, np.newaxis]
                             + post_scale
                             * post_trace[:, np.newaxis]
                             * pre_activities))

        return step_stdp


@Builder.register(STDP)
def build_stdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    post_trace = Signal(np.zeros(post_activities.size), name="post_trace")
    pre_scale = Signal(
        np.zeros(model.sig[conn]['weights'].shape), name="pre_scale")
    post_scale = Signal(
        np.zeros(model.sig[conn]['weights'].shape), name="post_scale")

    model.add_op(SimSTDP(pre_activities,
                         post_activities,
                         pre_trace,
                         post_trace,
                         pre_scale,
                         post_scale,
                         model.sig[conn]['weights'],
                         model.sig[rule]['delta'],
                         learning_rate=stdp.learning_rate,
                         pre_tau=stdp.pre_tau,
                         post_tau=stdp.post_tau,
                         pre_amp=stdp.pre_amp,
                         post_amp=stdp.post_amp,
                         bounds=stdp.bounds,
                         max_weight=stdp.max_weight,
                         min_weight=stdp.min_weight))

    # expose these for probes
    model.sig[rule]['pre_trace'] = pre_trace
    model.sig[rule]['post_trace'] = post_trace
    model.sig[rule]['pre_scale'] = pre_scale
    model.sig[rule]['post_scale'] = post_scale

    model.params[rule] = None  # no build-time info to return


class TripletSTDP(LearningRuleType):
    r"""Triplet spike-timing dependent plasticity rule.

    From "Triplets of Spikes in a Model of Spike Timing-Dependent Plasticity",
    Pfister & Gerstner, 2006. Unlike the doublet rule, the triplet rule
    models the interactions of triplets of spikes (pre-post-pre and
    post-pre-post patterns).

    The weight update rule is

    .. math::

       {d\omega_{ij} \over dt} = tr_{i1}(t) \sum_{sj} \delta (t-t_j^{sj})
         \left[ A_{2+} + A_{3+} tr_{j2}(t - \epsilon)\right]
         - tr_{j1}(t) \sum_{si} \delta (t-t_i^{si})
         \left[ A_{2-} + A_{3-} tr_{i2}(t - \epsilon)\right]

    where

    * :math:`A_{2+}`, :math:`A_{2-}`, :math:`A_{3+}`, :math:`A_{3-}` are
      positive amplitude constants
    * :math:`\epsilon` is a small constant, representing a time point just
      before a spike

    See :doc:`examples/stdp` for more details.

    Parameters
    ----------
    pre_tau : float
       Time constant on the first presynaptic trace decay.
    pre_taux : float
       Time constant on the second presynaptic trace decay.
    post_tau : float
       Time constant on the first postsynaptic trace decay.
    post_tauy : float
       Time constant on the second postsynaptic trace decay.
    pre_amp2 : float
       First presynaptic trace amplitude.
    pre_amp3 : float
       Second presynaptic trace amplitude.
    post_amp2 : float
       First postsynaptic trace amplitude.
    post_amp3 : float
       Second postsynaptic trace amplitude.
    nearest_spike : bool
       Whether traces should be set to ``amp / dt`` on every spike (``True``)
       or incremented by ``amp / dt`` for each spike (``False``, the default).
    """

    modifies = 'weights'
    probeable = ['pre_trace1', 'pre_trace2', 'post_trace1', 'post_trace2']

    learning_rate = NumberParam(
        'learning_rate', low=0, readonly=True, default=1e-9)
    pre_tau = NumberParam("pre_tau", low=0, low_open=True, default=0.0168)
    pre_taux = NumberParam("pre_taux", low=0, low_open=True, default=0.101)
    post_tau = NumberParam("post_tau", low=0, low_open=True, default=0.0337)
    post_tauy = NumberParam("post_tauy", low=0, low_open=True, default=0.125)
    pre_amp2 = NumberParam("pre_amp2", low=0, low_open=True, default=5e-10)
    pre_amp3 = NumberParam("pre_amp3", low=0, low_open=True, default=6.2e-3)
    post_amp2 = NumberParam("post_amp2", low=0, low_open=True, default=7e-3)
    post_amp3 = NumberParam("post_amp3", low=0, low_open=True, default=2.3e-4)
    nearest_spike = BoolParam("nearest_spike", default=False)

    def __init__(self, learning_rate=Default,
                 pre_tau=Default, pre_taux=Default,
                 post_tau=Default, post_tauy=Default,
                 pre_amp2=Default, pre_amp3=Default,
                 post_amp2=Default, post_amp3=Default, nearest_spike=Default):
        super(TripletSTDP, self).__init__(learning_rate, size_in=0)
        self.pre_tau = pre_tau
        self.pre_taux = pre_taux
        self.post_tau = post_tau
        self.post_tauy = post_tauy
        self.pre_amp2 = pre_amp2
        self.pre_amp3 = pre_amp3
        self.post_amp2 = post_amp2
        self.post_amp3 = post_amp3
        self.nearest_spike = nearest_spike

    def use(self, params):
        """Use a parameter set defined by Pfister & Gerstner, 2006.

        Parameters
        ----------
        params : str
            The parameter set to use; one of ``"visual"`` or ``"hippocampal"``,
            denoting the neuron types used for parameter matching.
        """
        if params == 'visual' and self.nearest_spike:
            self.pre_taux = 0.714
            self.post_tauy = 0.04
            self.pre_amp2 = 8.8e-11
            self.pre_amp3 = 5.3e-2
            self.post_amp2 = 6.6e-3
            self.post_amp3 = 3.1e-3
        elif params == 'visual' and not self.nearest_spike:
            self.pre_taux = 0.101
            self.post_tauy = 0.125
            self.pre_amp2 = 5e-10
            self.pre_amp3 = 6.2e-3
            self.post_amp2 = 7e-3
            self.post_amp3 = 2.3e-4
        elif params == 'hippocampal' and self.nearest_spike:
            self.pre_taux = 0.575
            self.post_tauy = 0.047
            self.pre_amp2 = 4.6e-3
            self.pre_amp3 = 9.1e-3
            self.post_amp2 = 3e-3
            self.post_amp3 = 7.5e-9
        elif params == 'hippocampal' and not self.nearest_spike:
            self.pre_taux = 0.946
            self.post_tauy = 0.027
            self.pre_amp2 = 6.1e-3
            self.pre_amp3 = 6.7e-3
            self.post_amp2 = 1.6e-3
            self.post_amp3 = 1.4e-3
        else:
            raise ValueError("Only 'visual' and 'hippocampal' recognized.")

        # Same for all parameter sets
        self.pre_tau = 0.0168
        self.post_tau = 0.0337


class SimTripletSTDP(Operator):
    def __init__(self, pre_activities, post_activities,
                 pre_trace1, post_trace1, pre_trace2, post_trace2,
                 delta, learning_rate,
                 pre_tau, pre_taux, post_tau, post_tauy,
                 pre_amp2, pre_amp3, post_amp2, post_amp3, nearest_spike):
        self.pre_activities = pre_activities
        self.post_activities = post_activities
        self.pre_trace1 = pre_trace1
        self.post_trace1 = post_trace1
        self.pre_trace2 = pre_trace2
        self.post_trace2 = post_trace2
        self.delta = delta
        self.learning_rate = learning_rate
        self.pre_tau = pre_tau
        self.pre_taux = pre_taux
        self.post_tau = post_tau
        self.post_tauy = post_tauy
        self.pre_amp2 = pre_amp2
        self.pre_amp3 = pre_amp3
        self.post_amp2 = post_amp2
        self.post_amp3 = post_amp3
        self.nearest_spike = nearest_spike

        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities]
        self.updates = [
            delta, pre_trace1, post_trace1, pre_trace2, post_trace2]

    def make_step(self, signals, dt, rng):
        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace1 = signals[self.pre_trace1]
        post_trace1 = signals[self.post_trace1]
        pre_trace2 = signals[self.pre_trace2]
        post_trace2 = signals[self.post_trace2]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        pre_t1 = dt / self.pre_tau
        post_t1 = dt / self.post_tau
        pre_t2 = dt / self.pre_taux
        post_t2 = dt / self.post_tauy

        if self.nearest_spike:

            def update_beforedelta():
                # Trace 1 gets full update
                pre_trace1[...] += -pre_trace1 * pre_t1
                post_trace1[...] += -post_trace1 * post_t1
                pre_s = pre_activities > 0
                pre_trace1[pre_s] = pre_activities[pre_s] * pre_t1
                post_s = post_activities > 0
                post_trace1[post_s] = post_activities[post_s] * post_t1

                # Trace 2 gets spike update later
                pre_trace2[...] += -pre_trace2 * pre_t2
                post_trace2[...] += -post_trace2 * post_t2

            def update_afterdelta():
                pre_s = pre_activities > 0
                pre_trace2[pre_s] = pre_activities[pre_s] * pre_t2
                post_s = post_activities > 0
                post_trace2[post_s] = post_activities[post_s] * post_t2
        else:

            def update_beforedelta():
                # Trace 1 gets full update
                pre_trace1[...] += pre_t1 * (-pre_trace1 + pre_activities)
                post_trace1[...] += post_t1 * (-post_trace1 + post_activities)
                # Trace 2 gets spike update later
                pre_trace2[...] += pre_t2 * -pre_trace2
                post_trace2[...] += post_t2 * -post_trace2

            def update_afterdelta():
                pre_trace2[...] += pre_t2 * pre_activities
                post_trace2[...] += post_t2 * post_activities

        def step_tripletstdp():
            # Update first traces before weight update
            update_beforedelta()

            delta[...] = (alpha
                          * (pre_trace1[np.newaxis, :]
                             * post_activities[:, np.newaxis]
                             * (self.pre_amp2
                                + self.pre_amp3 * post_trace2[:, np.newaxis])
                             - post_trace1[:, np.newaxis]
                             * pre_activities[np.newaxis, :]
                             * (self.post_amp2
                                + self.post_amp3 * pre_trace2[np.newaxis, :])))

            # Update second traces after weight update
            update_afterdelta()

        return step_tripletstdp


@Builder.register(TripletSTDP)
def build_tripletstdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    pre_trace1 = Signal(np.zeros(pre_activities.size), name="pre_trace1")
    post_trace1 = Signal(np.zeros(post_activities.size), name="post_trace1")
    pre_trace2 = Signal(np.zeros(pre_activities.size), name="pre_trace2")
    post_trace2 = Signal(np.zeros(post_activities.size), name="post_trace2")

    model.add_op(SimTripletSTDP(pre_activities,
                                post_activities,
                                pre_trace1,
                                post_trace1,
                                pre_trace2,
                                post_trace2,
                                model.sig[rule]['delta'],
                                learning_rate=stdp.learning_rate,
                                pre_tau=stdp.pre_tau,
                                pre_taux=stdp.pre_taux,
                                post_tau=stdp.post_tau,
                                post_tauy=stdp.post_tauy,
                                pre_amp2=stdp.pre_amp2,
                                pre_amp3=stdp.pre_amp3,
                                post_amp2=stdp.post_amp2,
                                post_amp3=stdp.post_amp3,
                                nearest_spike=stdp.nearest_spike))

    # expose these for probes
    model.sig[rule]['pre_trace1'] = pre_trace1
    model.sig[rule]['post_trace1'] = post_trace1
    model.sig[rule]['pre_trace2'] = pre_trace2
    model.sig[rule]['post_trace2'] = post_trace2

    model.params[rule] = None  # no build-time info to return
