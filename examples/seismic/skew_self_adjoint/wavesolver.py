from copy import copy
from devito import Function, TimeFunction
from devito.tools import memoized_meth
from examples.seismic import PointSource, Receiver
from examples.seismic.skew_self_adjoint.operators import (
    SSA_ISO_ForwardOperator, SSA_ISO_AdjointOperator,
    SSA_ISO_JacobianForwardOperator, SSA_ISO_JacobianAdjointOperator
)
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from examples.seismic.skew_self_adjoint import utils.*
from pyrevolve import Revolver


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
    def __init__(self, npad, qmin, qmax, b, v, src, rec, time_axis,
                 space_order=8, **kwargs):
        self.npad = npad
        self.qmin = qmin
        self.qmax = qmax
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
        
    # Note on use of memoized op_fwd, op_adj, op_jacobian_fwd, op_jacobian_adj:
    #   For repeated calls these functions only do the heavy lifting of building the
    #   Devito Operator if something 'impactful' changes in the symbolic representation. 

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, self.src, self.rec,
                               self.time_axis, space_order=self.space_order,
                               save=save, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for AdjointOperator runs"""
        return AdjointOperator(self.model, self.src, self.rec,
                               self.time_axis, space_order=self.space_order,
                               save=None, **self._kwargs)

    @memoized_meth
    def op_jacobian_fwd(self):
        """Cached operator for JacobianForward runs"""
        return JacobianForwardOperator(self.model, self.src, self.rec,
                                       self.time_axis, space_order=self.space_order,
                                       save=None, **self._kwargs)

    @memoized_meth
    def op_jacobian_adj(self, save=True):
        """Cached operator for JacobianAdjoint runs"""
        return JacobianAdjointOperator(self.model, self.src, self.rec,
                                       self.time_axis, space_order=self.space_order,
                                       save=save, **self._kwargs)

    
    def forward(self, src=None, rec=None, b=None, v=None, wOverQ=None, u=None, 
                save=None, **kwargs):
        """
        Forward modeling function that creates the necessary
        data objects for running a forward modeling operator.

        Parameters
        ----------
        src : SparseTimeFunction, optional, defaults to src at construction
            Time series data for the injected source term.
        rec : SparseTimeFunction, optional, defaults to rec at construction
            The interpolated receiver data.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        wOverQ : Function or float, optional, defaults to wOverQ at construction
            The time-constant dissipation only attenuation w/Q field.
        u : TimeFunction, optional
            Stores the computed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        -------
        Receiver time series data, wavefield TimeFunction u, and performance summary
        """
        # Get src. src cant change, use self.src if not passed
        src = src or self.src
        
        # Get rec. rec can change, create new if not passed
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
        model = { 'b': b, 'v': v, 'wOverQ': wOverQ }
        
        # Create the wavefield if not provided
        p = p or TimeFunction(name='u', grid=self.v.grid,
                              save=self.time_axis.num if save else None,
                              time_order=2, space_order=self.space_order)

        # Execute operator 
        summary = self.op_fwd(save).apply(model=model, src=src, rec=rec, u=u, **kwargs)
        return rec, u, summary

    
    def adjoint(self, src=None, rec=None, b=None, v=None, wOverQ=None, u=None, 
                save=None, **kwargs):
        """
        Adjoint modeling function that creates the necessary
        data objects for running a adjoint modeling operator.

        Parameters
        ----------
        src : SparseTimeFunction, optional, defaults to src at construction
            Time series data for the injected source term.
        rec : SparseTimeFunction, optional, defaults to rec at construction
            The interpolated receiver data.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        wOverQ : Function or float, optional, defaults to wOverQ at construction
            The time-constant dissipation only attenuation w/Q field.
        u : TimeFunction, optional
            Stores the computed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        -------
        Adjoint source time series data, wavefield TimeFunction u, and performance summary
        """
        # Get src. src can change, create new if not passed
        src = src or PointSource(name='src', grid=self.model.grid,
                                 time_range=self.time_axis,
                                 coordinates=self.src.coordinates)
        
        # Get rec. rec cant change, can use self.srec if not passed
        rec = rec or self.rec

        # Get (b, v, wOverQ) from passed arguments or from (b, v, wOverQ) at construction 
        b = b or self.b
        v = v or self.v
        wOverQ = wOverQ or self.wOverQ

        # ensure src, rec, b, v, wOverQ all share the same underlying grid
        assert src.grid == rec.grid == b.grid == v.grid == wOverQ.grid

        # Make dictionary of the physical model properties
        model = { 'b': b, 'v': v, 'wOverQ': wOverQ }
        
        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, v=v, vp=vp,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary

    
    def jacobian_forward(self, dmin, src=None, rec=None, u=None, U=None, vp=None, **kwargs):
        """
        Linearized JacobianForward modeling function that creates the necessary
        data objects for running an adjoint modeling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The forward wavefield.
        U : TimeFunction, optional
            The linearized wavefield.
        vp : Function or float, optional
            The time-constant velocity.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        summary = self.op_jacobian_forward().apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                       vp=vp, dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, U, summary

    
    def jacobian_adjoint(self, rec, u, v=None, grad=None, vp=None, checkpointing=False, **kwargs):
        """
        JacobianAdjoint modeling function for computing the adjoint of the
        Linearized JacobianForward modeling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the jacobian_forward field.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        JacobianAdjoint field and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        # JacobianAdjoint symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False), src=self.geometry.src,
                                         u=u, vp=vp, dt=dt)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          vp=vp, rec=rec, dt=dt, grad=grad)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_grad().apply(rec=rec, grad=grad, v=v, u=u, vp=vp,
                                           dt=dt, **kwargs)
        return grad, summary
