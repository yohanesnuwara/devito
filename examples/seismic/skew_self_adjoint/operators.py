from devito import Eq, Operator, Function, TimeFunction, Inc
from examples.seismic import PointSource, Receiver


def iso_stencil(field, b, v, wOverQ, **kwargs):
    """
    Stencil for the scalar isotropic visco- acoustic variable density
    skew self adjoint wave equation:

        b/v^2 [ P.dt2 + w/Q P.dt ] = (b P.dx).dx + (b P.dy).dy + (b P.dz).dz + s

    Note derivative shifts are omitted for simplicity above.
    See implementation notebook ssa_01_iso_implementation1.ipynb for more details.

    Parameters
    ----------
    field : TimeFunction, required
        The pressure wavefield computed solution.
    b : Function, required
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function, required
        Velocity (units: m/msec or km/sec)
    wOverQ : Function, required
        The w/Q field for dissipation only attenuation.
    forward : bool, optional
        The propagation direction. Defaults to True.
    q : TimeFunction, Function or float, optional
        Full-space/time source of the wave-equation.
    """

    # Define time step of pressure wavefield to be updated
    field_next = field.forward if kwargs.get('forward', True) else field.backward
    field_prev = field.backward if kwargs.get('forward', True) else field.forward

    # Get the source
    q = kwargs.get('q', 0)

    # Define the time update equation for 2d/3d
    # TODO: consider replacing conditional logic with
    # space_fd = sum([getattr(b * getattr(field, 'd%s'%d.name)(x0=d+d.spacing/2)),
    #            'd%s'%d.name)(x0=d-d.spacing/2)) for d in field.dimensions[1:]])
    if len(field.data.shape) == 3:
        t, x, z = field.dimensions
        eq_time_update = (t.spacing**2 * v**2 / b) * \
            ((b * field.dx(x0=x+x.spacing/2)).dx(x0=x-x.spacing/2) +
             (b * field.dz(x0=z+z.spacing/2)).dz(x0=z-z.spacing/2) + q) - \
            t.spacing * wOverQ * (field - field_prev) + 2 * field - field_prev

    else:
        t, x, y, z = field.dimensions
        eq_time_update = (t.spacing**2 * v**2 / b) * \
            ((b * field.dx(x0=x+x.spacing/2)).dx(x0=x-x.spacing/2) +
             (b * field.dy(x0=y+y.spacing/2)).dy(x0=y-y.spacing/2) +
             (b * field.dz(x0=z+z.spacing/2)).dz(x0=z-z.spacing/2) + q) - \
            t.spacing * wOverQ * (field - field_prev) + 2 * field - field_prev

    return [Eq(field_next, eq_time_update)]


def SSA_ISO_ForwardOperator(b, v, wOverQ, src, rec, time_axis, 
                            space_order=8, save=False, **kwargs):
    """
    Construct a forward modeling operator in a variable density visco- acoustic media.
    See implementation notebook ssa_01_iso_implementation1.ipynb for more details.

    Parameters
    ----------
    b : Function, required
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function, required
        Velocity (units: m/msec or km/sec)
    wOverQ : Function, required
        The w/Q field for dissipation only attenuation.
    src : SparseTimeFunction (PointSource)
        Source position and time signature.
    rec : SparseTimeFunction (PointSource)
        Receiver positions and time siganture.
    time_axis : TimeAxis 
        Defines temporal sampling
    space_order : int, optional, Defaults to 8
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    # Create symbols for wavefield, source and receivers
    p = TimeFunction(name='p', grid=v.grid,
                     save=time_axis.num if save else None,
                     time_order=2, space_order=space_order)

    # Time update equation
    eqn = iso_stencil(p, b, v, wOverQ, forward=True)

    # Construct expression to inject source values, injecting at p(t+dt)
    t = v.dimensions[0]
    src_term = src.inject(field=p.forward, expr=src * t.spacing**2 * v**2 / b)

    # Create interpolation expression for receivers, extracting at p(t)
    rec_term = rec.interpolate(expr=p)

    # Substitute spacing terms to reduce flops
    dt = time_axis.step
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing: dt})

    return Operator(eqn + src_term + rec_term, subs=spacing_map,
                    name='SSA_ISO_ForwardOperator', **kwargs)


def SSA_ISO_AdjointOperator(b, v, wOverQ, src, rec, time_axis
                            space_order=8, save=False, **kwargs):
    """
    Construct a adjoint modeling operator in a variable density visco- acoustic media.
    Note the FD evolution will be time reversed.
    See implementation notebook ssa_01_iso_implementation1.ipynb for more details.

    Parameters
    ----------
    b : Function, required
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function, required
        Velocity (units: m/msec or km/sec)
    wOverQ : Function, required
        The w/Q field for dissipation only attenuation.
    src : SparseTimeFunction (PointSource)
        Source position and time signature.
    rec : SparseTimeFunction (PointSource)
        Receiver positions and time siganture.
    time_axis : TimeAxis 
        Defines temporal sampling
    space_order : int, optional, Defaults to 8
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    # Create symbols for wavefield, source and receivers
    p = TimeFunction(name='p', grid=geometry.grid,
                          save=geometry.nt if save else None,
                          time_order=2, space_order=space_order)

    srca = PointSource(name='srca', grid=geometry.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Time update equation
    eqn = iso_stencil(p, b, v, wOverQ, forward="False")

    # Construct expression to inject receiver values, injecting at p(t-dt)
    t = v.dimensions[0]
    receivers = rec.inject(field=p.backward, expr=rec * t.spacing**2 * v**2 / b)

    # Create interpolation expression for the adjoint-source, extracting at p(t)
    source_a = srca.interpolate(expr=p)

    # Substitute spacing terms to reduce flops
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing: dt})

    return Operator(eqn + receivers + source_a, subs=spacing_map,
                    name='SSA_ISO_AdjointOperator', **kwargs)


def SSA_ISO_JacobianForwardOperator(b, v, wOverQ, src, rec, time_axis,
                                    space_order=8, save=False, **kwargs):
    """
    Construct a linearized JacobianForward modeling operator in a variable density
    visco- acoustic media.

    Parameters
    ----------
    b : Function, required
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function, required
        Velocity (units: m/msec or km/sec)
    wOverQ : Function, required
        The w/Q field for dissipation only attenuation.
    src : SparseTimeFunction (PointSource)
        Source position and time signature.
    rec : SparseTimeFunction (PointSource)
        Receiver positions and time siganture.
    time_axis : TimeAxis 
        Defines temporal sampling
    space_order : int, optional, Defaults to 8
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    # Create source and receiver symbols
    src = Receiver(name='src', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Create p0, dp wavefields and dv velocity perturbation field
    p0 = TimeFunction(name="p0", grid=geometry.grid,
                      save=geometry.nt if save else None,
                      time_order=2, space_order=space_order)

    dp = TimeFunction(name="dp", grid=geometry.grid,
                      save=geometry.nt if save else None,
                      time_order=2, space_order=space_order)

    dv = Function(name="dv", grid=geometry.grid, space_order=space_order)

    # Time update equations
    # JKW: this is pretty cool, simultaneously solving for p0 and dp!
    # The 1st equation is derived in ssa_01_iso_implementation1.ipynb
    # The 2nd equation is derived in ssa_02_iso_implementation2.ipynb
    t = v.dimensions[0]
    eqn1 = iso_stencil(p0, b, v, wOverQ, forward=True)
    eqn2 = iso_stencil(dp, b, v, wOverQ, forward=True,
                       q=2 * b * dv * v**-2 * (wOverQ * p0.dt(x0=t-t.spacing/2) + p0.dt2))

    # Construct expression to inject source values, injecting at p0(t+dt)
    src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * v**2 / b)

    # Create interpolation expression for receivers, extracting at dp(t)
    rec_term = rec.interpolate(expr=dp)

    # Substitute spacing terms to reduce flops
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing: dt})

    return Operator(eqn1 + src_term + eqn2 + rec_term, subs=spacing_map,
                    name='SSA_ISO_JacobianForwardOperator', **kwargs)


def SSA_ISO_JacobianAdjointOperator(b, v, wOverQ, src, rec, time_axis,
                                    space_order=8, save=True, **kwargs):
    """
    Construct a linearized JacobianAdjoint modeling operator in a variable density
    visco- acoustic media.

    Parameters
    ----------
    b : Function, required
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function, required
        Velocity (units: m/msec or km/sec)
    wOverQ : Function, required
        The w/Q field for dissipation only attenuation.
    src : SparseTimeFunction (PointSource)
        Source position and time signature.
    rec : SparseTimeFunction (PointSource)
        Receiver positions and time siganture.
    time_axis : TimeAxis 
        Defines temporal sampling
    space_order : int, optional, Defaults to 8
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    # Create receiver symbol
    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Create p0, dp wavefields and dv velocity perturbation field
    p0 = TimeFunction(name="p0", grid=geometry.grid,
                      save=geometry.nt if save else None,
                      time_order=2, space_order=space_order)

    dp = TimeFunction(name="dp", grid=geometry.grid,
                      save=geometry.nt if save else None,
                      time_order=2, space_order=space_order)

    dv = Function(name="dv", grid=geometry.grid, space_order=space_order)

    # Time update equation
    t = v.dimensions[0]
    eqn = iso_stencil(p0, b, v, wOverQ, forward=False)
    dv_update = Inc(dv, 2 * b * v**-3 * (wOverQ * p0.dt(x0=t-t.spacing/2) + p0.dt2))

    # Construct expression to inject receiver values, injecting at p(t-dt)
    rec_term = rec.inject(field=dp.backward, expr=rec * t.spacing**2 * v**2 / b)

    # Substitute spacing terms to reduce flops
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing: dt})

    return Operator(eqn + rec_term + [dv_update], subs=spacing_map,
                    name='SSA_ISO_JacobianAdjointOperator', **kwargs)
