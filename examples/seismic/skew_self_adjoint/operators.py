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
    field : TimeFunction
        The pressure wavefield computed solution.
    b : Function
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function
        Velocity (units: m/msec or km/sec)
    wOverQ : Function
        The w/Q field for dissipation only attenuation.
    forward : bool
        The propagation direction. Defaults to True.
    q : TimeFunction, Function or float
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


def ForwardOperator(b, v, wOverQ, geometry, dt, space_order=8,
                    save=False, **kwargs):
    """
    Construct a forward modeling operator in a variable density visco- acoustic media.
    See implementation notebook ssa_01_iso_implementation1.ipynb for more details.

    Parameters
    ----------
    b : Function
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function
        Velocity (units: m/msec or km/sec)
    wOverQ : Function
        The w/Q field for dissipation only attenuation.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional, Defaults to 8
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    # Create symbols for wavefield, source and receivers
    P = TimeFunction(name='P', grid=geometry.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)

    src = PointSource(name='src', grid=geometry.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # time update equation
    eqn = iso_stencil(P, b, v, wOverQ, forward="True")

    # Construct expression to inject source values, injecting at P(t+dt)
    t = v.dimensions[0]
    src_term = src.inject(field=P.forward, expr=src * t.spacing**2 * v**2 / b)

    # Create interpolation expression for receivers, extracting at P(t)
    rec_term = rec.interpolate(expr=P)

    # Substitute spacing terms to reduce flops
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing : dt})

    return Operator(eqn + src_term + rec_term, subs=spacing_map,
                    name='ForwardOperator', **kwargs)


def AdjointOperator(b, v, wOverQ, geometry, dt, space_order=8,
                    save=False, **kwargs):
    """
    Construct a adjoint modeling operator in a variable density visco- acoustic media.
    Note the FD evolution will be time reversed.
    See implementation notebook ssa_01_iso_implementation1.ipynb for more details.

    Parameters
    ----------
    b : Function
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function
        Velocity (units: m/msec or km/sec)
    wOverQ : Function
        The w/Q field for dissipation only attenuation.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional, Defaults to 8
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    # Create symbols for wavefield, source and receivers
    Ptilde = TimeFunction(name='Ptilde', grid=geometry.grid,
                          save=geometry.nt if save else None,
                          time_order=2, space_order=space_order)

    srca = PointSource(name='srca', grid=geometry.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    eqn = iso_stencil(Ptilde, b, v, wOverQ, forward="False")

    # Construct expression to inject receiver values, injecting at P(t-dt)
    t = v.dimensions[0]
    receivers = rec.inject(field=Ptilde.backward, expr=rec * t.spacing**2 * v**2 / b)

    # Create interpolation expression for the adjoint-source, extracting at P(t)
    source_a = srca.interpolate(expr=Ptilde)

    # Substitute spacing terms to reduce flops
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing : dt})

    return Operator(eqn + receivers + source_a, subs=spacing_map,
                    name='AdjointOperator', **kwargs)


def JacobianForwardOperator(b, v, wOverQ, geometry, dt, space_order=8,
                            save=False, **kwargs):
    """
    Construct a linearized JacobianForward modeling operator in a variable density
    visco- acoustic media.

    Parameters
    ----------
    b : Function
        Buoyancy = reciprocal density (units: m^3/kg)
    v : Function
        Velocity (units: m/msec or km/sec)
    wOverQ : Function
        The w/Q field for dissipation only attenuation.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
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

    # Create P0, dP wavefields and dv velocity perturbation field
    P0 = TimeFunction(name="P0", grid=geometry.grid, save=None,
                      time_order=2, space_order=space_order)
    
    dP = TimeFunction(name="dP", grid=geometry.grid, save=None,
                      time_order=2, space_order=space_order)
    
    dv = Function(name="dv", grid=geometry.grid, space_order=0)

    # JKW: this is pretty cool, simultaneously solving for P0 and dP!
    eqn1 = iso_stencil(P0, b, v, wOverQ, forward="True")
    eqn2 = iso_stencil(dP, b, v, wOverQ, forward="True", q=2 * b * v**-3 * P0.dt2)

    # Construct expression to inject source values, injecting at P0(t+dt)
    t = v.dimensions[0]
    src_term = src.inject(field=P0.forward, expr=src * t.spacing**2 * v**2 / b)

    # Create interpolation expression for receivers, extracting at dP(t)
    rec_term = rec.interpolate(expr=dP)

    # Substitute spacing terms to reduce flops
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing : dt})

    return Operator(eqn1 + src_term + eqn2 + rec_term, subs=spacing_map,
                    name='JacobianForward', **kwargs)


def JacobianAdjointOperator(b, v, wOverQ, geometry, dt, space_order=8, 
                            save=True, **kwargs):
    """
    Construct a linearized JacobianAdjoint modeling operator in a variable density
    visco- acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional, Defaults to 8
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    """
    m, damp = model.m, model.damp

    # JacobianAdjoint symbol and wavefield symbols
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, m, s, damp, kernel, forward=False)

    if kernel == 'OT2':
        gradient_update = Inc(grad, - u.dt2 * v)
    elif kernel == 'OT4':
        gradient_update = Inc(grad, - (u.dt2 + s**2 / 12.0 * u.biharmonic(m**(-2))) * v)
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Substitute spacing terms to reduce flops
    spacing_map = v.grid.spacing_map
    spacing_map.update({t.spacing : dt})

    return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
                    name='JacobianAdjoint', **kwargs)
