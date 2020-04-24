from timeit import default_timer as timer
import numpy as np
from devito import Grid, Function, SpaceDimension, Constant
from examples.seismic import RickerSource, Receiver, TimeAxis
from devito.builtins import gaussian_smooth

__all__ = ['critical_dt', 'setup_wOverQ', 'defaultSetupIso']


def critical_dt(v):
    """
    Determine the temporal sampling to satisfy CFL stability.
    This method replicates the functionality in the Model class.

    Parameters
    ----------
    v : Function
        velocity
    """
    coeff = 0.38 if len(v.grid.shape) == 3 else 0.42
    dt = v.dtype(coeff * np.min(v.grid.spacing) / (np.max(v.data)))
    return v.dtype("%.5e" % dt)


def setup_wOverQ(wOverQ, w, qmin, qmax, npad, sigma=None):
    """
    Initialise spatially variable w/Q field used to implement attenuation and
    absorb outgoing waves at the edges of the model. Uses an outer product
    via numpy.ogrid[:n1, :n2] to speed up loop traversal for 2d and 3d.
    TODO: stop wasting so much memory with 9 tmp arrays ...

    Parameters
    ----------
    wOverQ : Function
        The omega over Q field used to implement attenuation in the model,
        and the absorbing boundary condition for outgoing waves.
    w : float32
        center angular frequency, e.g. peak frequency of Ricker source wavelet
        used for modeling.
    qmin : float32
        Q value at the edge of the model. Typically set to 0.1 to strongly
        attenuate outgoing waves.
    qmax : float32
        Q value in the interior of the model. Typically set to 100 as a
        reasonable and physically meaningful Q value.
    npad : int
        Number of points in the absorbing boundary region. Note that we expect
        this to be the same on all sides of the model.
    sigma : float32
        sigma value for call to scipy gaussian smoother, default 5.
    """
    # sanity checks
    assert w > 0, "supplied w value [%f] must be positive" % (w)
    assert qmin > 0, "supplied qmin value [%f] must be positive" % (qmin)
    assert qmax > 0, "supplied qmax value [%f] must be positive" % (qmax)
    assert npad > 0, "supplied npad value [%f] must be positive" % (npad)
    for n in wOverQ.data.shape:
        if n - 2*npad < 1:
            raise ValueError("2 * npad must not exceed dimension size!")

    t1 = timer()
    sigma = sigma or npad//11
    lqmin = np.log(qmin)
    lqmax = np.log(qmax)

    if len(wOverQ.data.shape) == 2:
        # 2d operations
        nx, nz = wOverQ.data.shape
        kxMin, kzMin = np.ogrid[:nx, :nz]
        kxArr, kzArr = np.minimum(kxMin, nx-1-kxMin), np.minimum(kzMin, nz-1-kzMin)
        nval1 = np.minimum(kxArr, kzArr)
        nval2 = np.minimum(1, nval1/(npad))
        nval3 = np.exp(lqmin+nval2*(lqmax-lqmin))
        wOverQ.data[:, :] = w / nval3

    else:
        # 3d operations
        nx, ny, nz = wOverQ.data.shape
        kxMin, kyMin, kzMin = np.ogrid[:nx, :ny, :nz]
        kxArr = np.minimum(kxMin, nx-1-kxMin)
        kyArr = np.minimum(kyMin, ny-1-kyMin)
        kzArr = np.minimum(kzMin, nz-1-kzMin)
        nval1 = np.minimum(kxArr, np.minimum(kyArr, kzArr))
        nval2 = np.minimum(1, nval1/(npad))
        nval3 = np.exp(lqmin+nval2*(lqmax-lqmin))
        wOverQ.data[:, :, :] = w / nval3

    # Note if we apply the gaussian smoother, renormalize output to [qmin,qmax]
    if sigma > 0:
        nval2[:] = gaussian_smooth(nval3, sigma=sigma)
        nmin2, nmax2 = np.min(nval2), np.max(nval2)
        nval3[:] = qmin + (qmax - qmin) * (nval2 - nmin2) / (nmax2 - nmin2)

    wOverQ.data[:] = w / nval3

    # report min/max output Q value
    q1 = (np.min(1 / (wOverQ.data / w)))
    q2 = (np.max(1 / (wOverQ.data / w)))
    t2 = timer()
    print("setup_wOverQ ran in %.4f seconds -- min/max Q values; %.4f %.4f"
          % (t2-t1, q1, q2))


def defaultSetupIso(npad, shape, dtype, 
                    sigma=0, fpeak=0.010, qmin=0.1, qmax=100.0,
                    tmin=0.0, tmax=2000.0, bvalue=1.0/1000.0, vvalue=1.5):
    """
    For isotropic propagator build default model with 10m spacing,
        and 1.5 m/msec velocity

    Return:
        dictionary of velocity, buoyancy, and wOverQ
        TimeAxis defining temporal sampling
        SparseTimeFunction for source at (x=0, y=0, z=0)
        SparseTimeFunction for line of receivers at (x=[0:nr-1], y=ny//2, z=1)
    """
    d = 10.0
    origin = tuple([0.0 - d * npad for s in shape])
    extent = tuple([d * (s - 1) for s in shape])

    # Define dimensions
    if len(shape) == 2:
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
        grid = Grid(extent=extent, shape=shape, origin=origin, dimensions=(x, z), dtype=dtype)
    else:
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
        y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=d))
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
        grid = Grid(extent=extent, shape=shape, origin=origin, dimensions=(x, y, z), dtype=dtype)

    b = Function(name='b', grid=grid)
    v = Function(name='v', grid=grid)
    b.data[:] = bvalue
    v.data[:] = vvalue

    dt = critical_dt(v)
    time_axis = TimeAxis(start=tmin, stop=tmax, step=dt)

    nr = shape[0] - 2 * npad
    src = RickerSource(name='src', grid=grid, f0=fpeak, npoint=1, time_range=time_axis)
    rec = Receiver(name='rec', grid=grid, npoint=nr, time_range=time_axis)

    if len(shape) == 2:
        src.coordinates.data[0, 0] = 0.0
        src.coordinates.data[0, 1] = 0.0

        rec.coordinates.data[:, 0] = np.linspace(0.0, d * (nr - 1), nr)
        rec.coordinates.data[:, 1] = np.ones(nr, dtype=np.float32) * d
    else:
        src.coordinates.data[0, 0] = 0.0
        src.coordinates.data[0, 1] = 0.0
        src.coordinates.data[0, 2] = 0.0

        rec.coordinates.data[:, 0] = np.linspace(0.0, d * (nr - 1), nr)
        rec.coordinates.data[:, 1] = np.ones(nr, dtype=np.float32) * d * (shape[1]//2)
        rec.coordinates.data[:, 2] = np.ones(nr, dtype=np.float32) * d

    return b, v, time_axis, src, rec
