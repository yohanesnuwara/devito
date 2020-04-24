import numpy as np
import pytest
from devito import Grid, Function, TimeFunction, configuration
from examples.seismic import RickerSource, Receiver, TimeAxis
from examples.seismic.skew_self_adjoint import *
configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'

npad_default = 10
fpeak_default = 0.010
qmin_default = 0.1
qmax_default = 100.0

class TestWavesolver(object):

    # @pytest.mark.skip(reason="temporarily skip")
    # @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
    @pytest.mark.parametrize('shape', [(41, 51), ])
    @pytest.mark.parametrize('dtype', [np.float32, ])
    def test_adjointF(self, shape, dtype):
        """
        Test the linear modeling operator by verifying dot product test:
            r . F[m] s = s . F[m]^\t r
        """
        tol = 10 * np.finfo(dtype).eps
        omega = 2.0 * np.pi * fpeak_default
        v, b, time_axis, src, rec = defaultSetupIso(npad_default, shape, dtype)
        solver = SSA_ISO_AcousticWaveSolver(npad_default, qmin_default, qmax_default,
                                            omega, b, v, src, rec, time_axis)
        rec, u, summary = solver.forward()

