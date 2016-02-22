import nengo
import lasagne as lgn

from simulator import Simulator

nonlinearity_map = {nengo.RectifiedLinear: lgn.nonlinearities.rectify,
                    nengo.Sigmoid: lgn.nonlinearities.sigmoid,
                    nengo.Direct: lgn.nonlinearities.linear}
