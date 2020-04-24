from devito import Function, TimeFunction
from devito.tools import memoized_meth
from examples.seismic import PointSource, Receiver
from examples.seismic.skew_self_adjoint.utils import *
from examples.seismic.skew_self_adjoint.operators import *


class SSA_ISO_AcousticWaveSolver(object):
    """
    Solver object for a scalar isotropic variable density visco- acoustic skew
    self adjoint wave equation that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem setup.

    Parameters
    ----------
    npad : int, required
        Number of points in the absorbing boundary
        Typically set to 50
    omega : float, required
        Center circular frequency for dissipation only attenuation
    qmin : float, required
        Minimum Q value on the exterior of the absorbing boundary
        Tupically set to 0.1
    qmax : float, required
        Maximum Q value in the interior of the model
        Tupically set to 100.0
    b : Function, required
        Physical model with buoyancy (m^3/kg)
    v : Function, required
        Physical model with velocity (m/msec)
    src : SparseTimeFunction (PointSource)
        Source position and time signature.
    rec : SparseTimeFunction (PointSource)
        Receiver positions and time siganture.
    time_axis : TimeAxis
        Defines temporal sampling
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 8.
    """

    def __init__(self, npad, qmin, qmax, omega, b, v, src, rec, time_axis,
                 space_order=8, **kwargs):
        self.npad = npad
        self.qmin = qmin
        self.qmax = qmax
        self.omega = omega
        self.b = b
        self.v = v
        self.src = src
        self.rec = rec
        self.time_axis = time_axis
        self.space_order = space_order

        # Determine temporal sampling using critical_dt in utils.py
        self.dt = critical_dt(v)

        # Cache compiler options
        self._kwargs = kwargs

        # Create the wOverQ Function
        wOverQ = Function(name='wOverQ', grid=v.grid, space_order=v.space_order)
        setup_wOverQ(wOverQ, omega, qmin, qmax, npad)
        self.wOverQ = wOverQ

        # initialize dictionary to store model parameters
        self.model = { 'b': b, 'v': v, 'wOverQ': wOverQ }

    # Note on use of memoized op_fwd, op_adj, op_jacobian_fwd, op_jacobian_adj:
    #   For repeated calls these functions only do the heavy lifting of building the
    #   Devito Operator if something 'impactful' changes in the symbolic representation.
    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ISO_FwdOperator(self.model, self.src, self.rec,
                               self.time_axis, space_order=self.space_order,
                               save=save, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for AdjOperator runs"""
        return ISO_AdjOperator(self.model, self.src, self.rec,
                               self.time_axis, space_order=self.space_order,
                               save=None, **self._kwargs)

    @memoized_meth
    def op_jacobian_fwd(self):
        """Cached operator for JacobianForward runs"""
        return ISO_JacobianFwdOperator(self.model, self.src, self.rec,
                                       self.time_axis, space_order=self.space_order,
                                       save=None, **self._kwargs)

    @memoized_meth
    def op_jacobian_adj(self, save=True):
        """Cached operator for JacobianAdjoint runs"""
        return ISO_JacobianAdjOperator(self.model, self.rec,
                                       self.time_axis, space_order=self.space_order,
                                       save=save, **self._kwargs)

    def forward(self, src=None, rec=None, b=None, v=None, wOverQ=None, u=None,
                save=None, **kwargs):
        """
        Forward modeling function that creates the necessary
        data objects for running a forward modeling operator.
        No required parameters.

        Parameters
        ----------
        src : SparseTimeFunction, optional, defaults to src at construction
            Time series data for the injected source term.
        rec : SparseTimeFunction, optional, defaults to new rec
            The interpolated receiver data.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        wOverQ : Function or float, optional, defaults to wOverQ at construction
            The time-constant dissipation only attenuation w/Q field.
        u : Function or float, optional, defaults to new TimeFunction
            Stores the computed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Receiver time series data, TimeFunction wavefield u, and performance summary
        """
        # Get src: src cant change, use self.src if not passed
        src = src or self.src

        # Get rec: rec can change, create new if not passed
        rec = rec or Receiver(name='rec', grid=self.v.grid,
                              time_range=self.time_axis,
                              coordinates=self.rec.coordinates)

        # Get (b, v, wOverQ) from passed arguments or from (b, v, wOverQ) at construction
        b = b or self.b
        v = v or self.v
        wOverQ = wOverQ or self.wOverQ

        # ensure src, rec, b, v, wOverQ all share the same underlying grid
        assert src.grid == rec.grid == b.grid == v.grid == wOverQ.grid

        # Make dictionary of the physical model properties
        self.model.update(b=b, v=v, wOverQ=wOverQ)

        # Create the wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.v.grid,
                              save=self.time_axis.num if save else None,
                              time_order=2, space_order=self.space_order)

        # Execute operator, "splatting" the model dictionary entries
        summary = self.op_fwd(save).apply(**self.model, src=src, rec=rec, u=u, **kwargs)
        return rec, u, summary

    def adjoint(self, rec, srca=None, b=None, v=None, wOverQ=None, ua=None,
                save=None, **kwargs):
        """
        Adjoint modeling function that creates the necessary
        data objects for running a adjoint modeling operator.
        Required parameters: rec.

        Parameters
        ----------
        rec : SparseTimeFunction, required
            The interpolated receiver data to be injected
        srca : SparseTimeFunction, optional, defaults to new srca
            Time series data for the adjoint source term.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        wOverQ : Function or float, optional, defaults to wOverQ at construction
            The time-constant dissipation only attenuation w/Q field.
        ua : Function or float, optional, defaults to new TimeFunction
            Stores the computed adjoint wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Adjoint source time series data, wavefield TimeFunction ua,
        and performance summary
        """
        # Get src: src can change, create new if not passed
        srca = srca or PointSource(name='srca', grid=self.v.grid,
                                   time_range=self.time_axis,
                                   coordinates=self.src.coordinates)

        # Get (b, v, wOverQ) from passed arguments or from (b, v, wOverQ) at construction
        b = b or self.b
        v = v or self.v
        wOverQ = wOverQ or self.wOverQ

        # ensure srca, rec, b, v, wOverQ all share the same underlying grid
        assert srca.grid == rec.grid == b.grid == v.grid == wOverQ.grid

        # Make dictionary of the physical model properties
        self.model.update(b=b, v=v, wOverQ=wOverQ)

        # Create the adjoint wavefield if not provided
        ua = ua or TimeFunction(name='ua', grid=self.v.grid,
                                save=self.time_axis.num if save else None,
                                time_order=2, space_order=self.space_order)

        # Execute operator, "splatting" the model dictionary entries
        summary = self.op_adj().apply(**self.model, srca=srca, rec=rec, u=ua, **kwargs)
        return srca, ua, summary

    def jacobian_forward(self, dm, src=None, rec=None, b=None, v=None, wOverQ=None,
                         u0=None, du=None, save=None, **kwargs):
        """
        Linearized JacobianForward modeling function that creates the necessary
        data objects for running a Jacobian forward modeling operator.
        Required parameters: dm.

        Parameters
        ----------
        dm : Function or float, required
            The perturbation to the velocity model.
        src : SparseTimeFunction, optional, defaults to src at construction
            Time series data for the injected source term.
        rec : SparseTimeFunction, optional, defaults to new rec
            The interpolated receiver data.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        wOverQ : Function or float, optional, defaults to wOverQ at construction
            The time-constant dissipation only attenuation w/Q field.
        u0 : Function or float, optional, defaults to new TimeFunction
            Stores the computed background wavefield.
        du : Function or float, optional, defaults to new TimeFunction
            Stores the computed perturbed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Receiver time series data rec, TimeFunction background wavefield u0,
        TimeFunction perturbation wavefield du, and performance summary
        """
        # Get src: src cant change, use self.src if not passed
        src = src or self.src

        # Get rec: rec can change, create new if not passed
        rec = rec or Receiver(name='rec', grid=self.v.grid,
                              time_range=self.time_axis,
                              coordinates=self.rec.coordinates)

        # Get (b, v, wOverQ) from passed arguments or from (b, v, wOverQ) at construction
        b = b or self.b
        v = v or self.v
        wOverQ = wOverQ or self.wOverQ

        # ensure src, rec, b, v, wOverQ all share the same underlying grid
        assert src.grid == rec.grid == b.grid == v.grid == wOverQ.grid

        # Make dictionary of the physical model properties
        self.model.update(b=b, v=v, wOverQ=wOverQ)

        # Create the wavefields if not provided
        u0 = u0 or TimeFunction(name='u0', grid=self.v.grid,
                                save=self.time_axis.num if save else None,
                                time_order=2, space_order=self.space_order)

        du = du or TimeFunction(name='du', grid=self.v.grid,
                                save=self.time_axis.num if save else None,
                                time_order=2, space_order=self.space_order)

        # Execute operator, "splatting" the model dictionary entries
        summary = self.op_jacobian_fwd().apply(dm=dm, **self.model, src=src, rec=rec,
                                                    u0=u0, du=du, **kwargs)
        return rec, u0, du, summary

    def jacobian_adjoint(self, rec, u0, b=None, v=None, wOverQ=None,
                         dm=None, du=None, save=None, **kwargs):
        """
        Linearized JacobianForward modeling function that creates the necessary
        data objects for running a Jacobian forward modeling operator.
        Required parameters: rec, u0.

        Parameters
        ----------
        rec : SparseTimeFunction, required
            The interpolated receiver data to be injected.
        u0 : Function or float, required, (created with save=True)
            Stores the computed background wavefield.
        src : SparseTimeFunction, optional, defaults to src at construction
            Time series data for the injected source term.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        wOverQ : Function or float, optional, defaults to wOverQ at construction
            The time-constant dissipation only attenuation w/Q field.
        dm : Function or float, optional, defaults to new Function
            The perturbation to the velocity model.
        du : Function or float, optional, defaults to new TimeFunction
            Stores the computed perturbed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Function model perturbation dm, Receiver time series data rec,
        TimeFunction background wavefield u0, TimeFunction perturbation wavefield du,
        and performance summary
        """
        # Get model perturbation Function or create
        dm = dm or Function(name='dm', grid=self.v.grid, space_order=self.space_order)

        # Get (b, v, wOverQ) from passed arguments or from (b, v, wOverQ) at construction
        b = b or self.b
        v = v or self.v
        wOverQ = wOverQ or self.wOverQ

        # ensure src, rec, b, v, wOverQ all share the same underlying grid
        assert src.grid == rec.grid == b.grid == v.grid == wOverQ.grid

        # Make dictionary of the physical model properties
        self.model.update(b=b, v=v, wOverQ=wOverQ)

        # Create the perturbation wavefield if not provided
        du = du or TimeFunction(name='du', grid=self.v.grid,
                                save=self.time_axis.num if save else None,
                                time_order=2, space_order=self.space_order)

        # Execute operator, "splatting" the model dictionary entries
        summary = self.op_jacobian_adj(save).apply(dm=dm, **self.model, rec=rec,
                                                          u0=u0, du=du, **kwargs)
        return dm, u0, du, summary
