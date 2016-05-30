from __future__ import absolute_import

from keras.layers import Layer


class SoftLIF(Layer):
    def __init__(self, sigma=1., amplitude=1., tau_rc=0.02, tau_ref=0.002,
                 **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.amplitude = amplitude
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        super(SoftLIF, self).__init__(**kwargs)

    def call(self, x, mask=None):
        from keras import backend as K
        j = K.softplus((x - 1) / self.sigma) * self.sigma
        v = self.amplitude / (self.tau_ref + self.tau_rc*K.log(1 + 1/j))
        return K.switch(j > 0, v, 0)

    def get_config(self):
        config = {'sigma': self.sigma, 'amplitude': self.amplitude,
                  'tau_rc': self.tau_rc, 'tau_ref': self.tau_ref}
        base_config = super(SoftLIF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
