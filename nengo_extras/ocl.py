import mako
import nengo_ocl
import numpy as np
import pyopencl as cl


def plan_aml_decode(queue, pre, base_decoders, decoded, tag=None):
    assert pre.ctype == base_decoders.ctype == decoded.ctype
    assert len(pre) == len(base_decoders) == len(decoded)
    assert np.all(pre.shape0s == base_decoders.shape1s)
    assert np.all(base_decoders.shape0s == decoded.shape0s)

    text = """
    __kernel void aml_decode(
        __global const int *ds,
        __global const int *ns,
        __global const int *pre_stride0s,
        __global const int *pre_starts,
        __global const ${type} *pre_data,
        __global const int *base_decoders_stride0s,
        __global const int *base_decoders_starts,
        __global const ${type} *base_decoders_data,
        __global const int *decoded_stride0s,
        __global const int *decoded_starts,
        __global ${type} *decoded_data
    ) {
        const int i = get_global_id(0);
        const int k = get_global_id(1);

        const int d = ds[k];
        const int n = ns[k];

        __global const ${type} *pre = pre_data + pre_starts[k];
        __global const ${type} *base_decoders = base_decoders_data +
            base_decoders_starts[k];
        __global ${type} *decoded = decoded_data + decoded_starts[k];

        if (i < n) {
            ${type} x = 0.;
            for (int s = 0; s < d; ++s) {
                x += base_decoders[i * base_decoders_stride0s[k] + s] * pre[s];
            }
            decoded[i] = x;
        }
    }
    """

    textconf = dict(type=pre.ctype)
    text = nengo_ocl.utils.as_ascii(
        mako.template.Template(text, output_encoding="ascii").render(**textconf)
    )

    full_args = (
        base_decoders.cl_shape1s,
        base_decoders.cl_shape0s,
        pre.cl_stride0s,
        pre.cl_starts,
        pre.cl_buf,
        base_decoders.cl_stride0s,
        base_decoders.cl_starts,
        base_decoders.cl_buf,
        decoded.cl_stride0s,
        decoded.cl_starts,
        decoded.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().aml_decode
    _fn.set_args(*(arr.data for arr in full_args))

    lsize = None
    gsize = (base_decoders.shape0s.max(), len(pre))
    plan = nengo_ocl.plan.Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_aml_decode", tag=tag
    )
    plan.full_args = full_args  # prevent garbage collection
    plan.flops_per_call = np.sum(
        base_decoders.shape0s * base_decoders.shape1s * 2 + base_decoders.shape1s * 2
    )
    plan.bw_per_call = decoded.nbytes + pre.nbytes + base_decoders.nbytes

    return plan


def plan_aml(queue, error, decoders, delta, alpha, decoded, tag=None):
    assert error.ctype == decoders.ctype == alpha.ctype == decoded.ctype
    assert len(error) == len(decoders) == len(alpha) == len(decoded)
    assert np.all(error.shape0s - 2 == decoders.shape0s)

    text = """
    __kernel void aml(
        __global const int *ds,
        __global const int *ns,
        __global const int *error_stride0s,
        __global const int *error_starts,
        __global const ${type} *error_data,
        __global const int *decoders_stride0s,
        __global const int *decoders_starts,
        __global const ${type} *decoders_data,
        __global const int *delta_stride0s,
        __global const int *delta_starts,
        __global ${type} *delta_data,
        __global const int *decoded_stride0s,
        __global const int *decoded_starts,
        __global const ${type} *decoded_data,
        __global const ${type} *alphas
    ) {
        const int ij = get_global_id(0);
        const int k = get_global_id(1);

        const int d = ds[k];
        const int n = ns[k];
        const int i = ij / n;
        const int j = ij % n;

        __global const ${type} *decoders = decoders_data + decoders_starts[k];
        __global ${type} *delta = delta_data + delta_starts[k];
        const ${type} scale = error_data[error_starts[k]];
        const ${type} decay = error_data[error_starts[k] + 1];
        const ${type} error = error_data[error_starts[k] + i + 2];
        const ${type} decoded = decoded_data[decoded_starts[k] + j];
        const ${type} alpha = alphas[k];

        if (i < d) {
            delta[i * delta_stride0s[k] + j] =
                alpha * scale * error * decoded +
                decoders[i * decoders_stride0s[k] + j] * (decay - 1.);
        }
    }
    """

    textconf = dict(type=error.ctype)
    text = nengo_ocl.utils.as_ascii(
        mako.template.Template(text, output_encoding="ascii").render(**textconf)
    )

    full_args = (
        decoders.cl_shape0s,
        decoders.cl_shape1s,
        error.cl_stride0s,
        error.cl_starts,
        error.cl_buf,
        decoders.cl_stride0s,
        decoders.cl_starts,
        decoders.cl_buf,
        delta.cl_stride0s,
        delta.cl_starts,
        delta.cl_buf,
        decoded.cl_stride0s,
        decoded.cl_starts,
        decoded.cl_buf,
        alpha,
    )
    _fn = cl.Program(queue.context, text).build().aml
    _fn.set_args(*(arr.data for arr in full_args))

    lsize = None
    gsize = (decoders.sizes.max(), len(error))
    plan = nengo_ocl.plan.Plan(queue, _fn, gsize, lsize=lsize, name="cl_aml", tag=tag)
    plan.full_args = full_args  # prevent garbage collection
    plan.flops_per_call = np.sum(2 * (error.shape0s * decoded.shape0s))
    plan.bw_per_call = decoded.nbytes + error.nbytes + alpha.nbytes + decoders.nbytes

    return plan


class AmlSimulator(nengo_ocl.Simulator):
    def plan_SimAML(self, ops):
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        base_decoders = self.RaggedArray(
            [op.base_decoders for op in ops], dtype=np.float32
        )
        pre = self.all_data[[self.sidx[op.pre] for op in ops]]
        error = self.all_data[[self.sidx[op.error] for op in ops]]
        decoders = self.all_data[[self.sidx[op.decoders] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        decoded = self.RaggedArray(
            [np.zeros(op.decoders.shape[1]) for op in ops], dtype=np.float32
        )
        return [
            plan_aml_decode(self.queue, pre, base_decoders, decoded),
            plan_aml(self.queue, error, decoders, delta, alpha, decoded),
        ]
