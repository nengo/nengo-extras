import warnings

import numpy as np

from nengo.builder import Builder, Signal
from nengo.builder.operator import DotInc, ElementwiseInc, Reset, SimPyFunc
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleType
from nengo.params import EnumParam, FunctionParam, NumberParam
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

    The update is given by:

        \delta W_ij = \eta a_j e_i f(u_i)

    where ``e_i`` is the input error in the postsynaptic neuron space,
    ``a_j`` is the jth presynaptic neuron (output) activity,
    ``u_i`` is the ith postsynaptic neuron input,
    and ``f`` is a provided function.

    Parameters
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    pre_tau : float
        Filter constant on the presynaptic output ``a_j``.
    post_fn : callable
        Function ``f`` to apply to the postsynaptic inputs ``u_i``. The
        default of ``None`` means the ``f(u_i)`` term is omitted.
    post_tau : float
        Filter constant on the postsynaptic input ``u_i``. This defaults to
        ``None`` because these should typically be filtered by the connection.
    """
    modifies = 'weights'
    probeable = ('delta', 'in', 'error', 'correction', 'pre', 'post')

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True, optional=True)
    post_fn = DeltaRuleFunctionParam('post_fn', optional=True)
    post_target = EnumParam('post_target', values=('in', 'out'))

    def __init__(self, learning_rate=1e-4, pre_tau=0.005,
                 post_fn=None, post_tau=None, post_target='in'):
        if learning_rate >= 1.0:
            warnings.warn("This learning rate is very high, and can result "
                          "in floating point errors from too much current.")
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.post_fn = post_fn
        self.post_target = post_target
        super(DeltaRule, self).__init__(learning_rate, size_in='post')

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
