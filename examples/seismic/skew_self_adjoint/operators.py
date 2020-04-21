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
        t,x,z = field.dimensions
        eq_time_update = (t.spacing**2 * v**2 / b) * \
            ((b * field.dx(x0=x+x.spacing/2)).dx(x0=x-x.spacing/2) + \
             (b * field.dz(x0=z+z.spacing/2)).dz(x0=z-z.spacing/2) + q) - \
            t.spacing * wOverQ * (field - field_prev) + 2 * field - field_prev
    else:
        t,x,y,z = field.dimensions
        eq_time_update = (t.spacing**2 * v**2 / b) * \
            ((b * field.dx(x0=x+x.spacing/2)).dx(x0=x-x.spacing/2) + \
             (b * field.dy(x0=y+y.spacing/2)).dy(x0=y-y.spacing/2) + \
             (b * field.dz(x0=z+z.spacing/2)).dz(x0=z-z.spacing/2) + q) - \
            t.spacing * wOverQ * (field - field_prev) + 2 * field - field_prev
    
    return [Eq(field_next, eq_time_update)]


def ForwardOperator(model, geometry, space_order=8, save=False, **kwargs):
    """
    Construct a forward modelling operator in an variable density 
    visco- acoustic media.
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

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=b.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)

    src = PointSource(name='src', grid=geometry.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, b, v, wOverQ)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    
    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m, damp = model.m, model.damp

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, m, s, damp, kernel, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def JacobianAdjointOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    kernel : str, optional
        Type of discretization, centered or shifted.
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
    return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
                    name='JacobianAdjoint', **kwargs)


def JacobianForwardOperator(model, geometry, space_order=4,
                 kernel='OT2', **kwargs):
    """
    Construct an Linearized JacobianForward operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m, damp = model.m, model.damp

    # Create source and receiver symbols
    src = Receiver(name='src', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, m, s, damp, kernel)
    eqn2 = iso_stencil(U, m, s, damp, kernel, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name='JacobianForward', **kwargs)
