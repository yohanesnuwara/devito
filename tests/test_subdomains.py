import pytest
import numpy as np

from math import floor
import numpy as np
import pytest

from conftest import skipif
from devito import (Grid, Function, TimeFunction, Eq, solve, Operator, SubDomain,
                    SubDomainSet, Dimension)
from devito.data import LEFT, RIGHT

pytestmark = skipif(['yask', 'ops'])


class TestSubdomains(object):
    """
    Class for testing SubDomains
    """

    def test_multiple_middle(self):
        """
        Test Operator with two basic 'middle' subdomains defined.
        """
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()

        class sd1(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 6, 1), y: ('middle', 1, 1)}
        s_d1 = sd1()

        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1))

        f = Function(name='f', grid=grid, dtype=np.int32)

        eq0 = Eq(f, f+1, subdomain=grid.subdomains['d0'])
        eq1 = Eq(f, f+2, subdomain=grid.subdomains['d1'])

        Operator([eq0, eq1])()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        assert((np.array(f.data) == expected).all())

    def test_shape(self):
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()

        class sd1(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('right', 4), y: ('left', 2)}
        s_d1 = sd1()

        class sd2(SubDomain):
            name = 'd2'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('left', 3), y: ('middle', 1, 2)}
        s_d2 = sd2()

        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1, s_d2))

        assert grid.subdomains['domain'].shape == (10, 10)
        assert grid.subdomains['interior'].shape == (8, 8)

        assert grid.subdomains['d0'].shape == (3, 8)
        assert grid.subdomains['d1'].shape == (4, 2)
        assert grid.subdomains['d2'].shape == (3, 7)

    def test_iterate_NDomains(self):
        """
        Test that a set of subdomains are iterated upon correctly.
        """

        n_domains = 10

        class Inner(SubDomainSet):
            name = 'inner'

        bounds_xm = np.zeros((n_domains,), dtype=np.int32)
        bounds_xM = np.zeros((n_domains,), dtype=np.int32)
        bounds_ym = np.zeros((n_domains,), dtype=np.int32)
        bounds_yM = np.zeros((n_domains,), dtype=np.int32)

        for j in range(0, n_domains):
            bounds_xm[j] = j
            bounds_xM[j] = n_domains-1-j
            bounds_ym[j] = floor(j/2)
            bounds_yM[j] = floor(j/2)

        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        inner_sd = Inner(N=n_domains, bounds=bounds)

        grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

        f = TimeFunction(name='f', grid=grid, dtype=np.int32)
        f.data[:] = 0

        stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
                     subdomain=grid.subdomains['inner'])

        op = Operator(stencil)
        op(time_m=0, time_M=9, dt=1)
        result = f.data[0]

        expected = np.zeros((10, 10), dtype=np.int32)
        for j in range(0, n_domains):
            expected[j, bounds_ym[j]:n_domains-bounds_yM[j]] = 10

        assert((np.array(result) == expected).all())

    def test_multi_eq(self):
        """
        Test SubDomainSet functionality when multiple equations are
        present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        class MySubdomains(SubDomainSet):
            name = 'mydomains'

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = 1
        bounds_yM = 1
        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)
        my_sd = MySubdomains(N=n_domains, bounds=bounds)
        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd, ))

        assert(grid.subdomains['mydomains'].shape == ((3, 8), (3, 8)))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)
        h = Function(name='h', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains'])
        eq2 = Eq(g, g+1)
        eq3 = Eq(h, h+2, subdomain=grid.subdomains['mydomains'])

        op = Operator([eq1, eq2, eq3])
        op.apply()

        expected1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        expected2 = np.full((10, 10), 1, dtype=np.int32)
        expected3 = 2*expected1

        assert((np.array(f.data) == expected1).all())
        assert((np.array(g.data) == expected2).all())
        assert((np.array(h.data) == expected3).all())

    def test_multi_sets(self):
        """
        Check functionality for when multiple subdomain sets are present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        n = Dimension(name='n')
        m = Dimension(name='m')

        class MySubdomains1(SubDomainSet):
            name = 'mydomains1'
            implicit_dimension = n

        class MySubdomains2(SubDomainSet):
            name = 'mydomains2'
            implicit_dimension = m

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(1)
        bounds_yM = int(Ny/2+1)
        bounds1 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(Ny/2+1)
        bounds_yM = int(1)
        bounds2 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        my_sd1 = MySubdomains1(N=n_domains, bounds=bounds1)
        my_sd2 = MySubdomains2(N=n_domains, bounds=bounds2)

        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd1, my_sd2))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains1'])
        eq2 = Eq(g, g+2, subdomain=grid.subdomains['mydomains2'])

        op = Operator([eq1, eq2])
        op.apply()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

        assert((np.array(f.data[:]+g.data[:]) == expected).all())

    @pytest.mark.parallel(mode=4)
    def test_subdomainset_mpi(self):

        n_domains = 5

        class Inner(SubDomainSet):
            name = 'inner'

        bounds_xm = np.zeros((n_domains,), dtype=np.int32)
        bounds_xM = np.zeros((n_domains,), dtype=np.int32)
        bounds_ym = np.zeros((n_domains,), dtype=np.int32)
        bounds_yM = np.zeros((n_domains,), dtype=np.int32)

        for j in range(0, n_domains):
            bounds_xm[j] = j
            bounds_xM[j] = j
            bounds_ym[j] = j
            bounds_yM[j] = 2*n_domains-1-j

        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        inner_sd = Inner(N=n_domains, bounds=bounds)

        grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

        assert(grid.subdomains['inner'].shape == ((10, 1), (8, 1), (6, 1),
                                                  (4, 1), (2, 1)))

        f = TimeFunction(name='f', grid=grid, dtype=np.int32)
        f.data[:] = 0

        stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
                     subdomain=grid.subdomains['inner'])

        op = Operator(stencil)
        op(time_m=0, time_M=9, dt=1)
        result = f.data[0]

        fex = Function(name='fex', grid=grid)
        expected = np.zeros((10, 10), dtype=np.int32)
        for j in range(0, n_domains):
            expected[j, j:10-j] = 10
        fex.data[:] = np.transpose(expected)

        assert((np.array(result) == np.array(fex.data[:])).all())

    def test_func_allocation(self):

        class SubDomain_Left(SubDomain):
            name = 'sd_left'

            def define(self, dimensions):
                x, y = dimensions
                # Create 2 points in the left of x and y dimensions
                return {x: ('left', 2), y: ('left', 2)}

        class SubDomain_Middle(SubDomain):
            name = 'sd_middle'

            def define(self, dimensions):
                x, y = dimensions
                # Create a 2x2 subdomain in the middle of the grid.
                return {x: ('middle', 4, 4), y: ('middle', 4, 4)}

        class SubDomain_Right(SubDomain):
            name = 'sd_right'

            def define(self, dimensions):
                x, y = dimensions
                # Create a 2x2 subdomain in the middle of the grid.
                return {x: ('right', 2), y: ('right', 2)}

        sd_left = SubDomain_Left()
        sd_middle = SubDomain_Middle()
        sd_right = SubDomain_Right()

        grid = Grid((10, 10), subdomains=(sd_left, sd_middle, sd_right))

        u1 = Function(name='u1', grid=grid, subdomain=sd_left)
        u2 = Function(name='u2', grid=grid, subdomain=sd_middle)
        u3 = Function(name='u3', grid=grid, subdomain=sd_right)
        v = Function(name='v', grid=grid)

        # All u Functions is defined as a 2x2 grid
        assert u1.shape == (2, 2)
        assert u2.shape == (2, 2)
        assert u3.shape == (2, 2)

        # Apply equations to each subdomain
        eqn1 = Eq(v, u1 + 1, subdomain=grid.subdomains['sd_left'])
        eqn2 = Eq(v, u2 + 2, subdomain=grid.subdomains['sd_middle'])
        eqn3 = Eq(v, u2 + 3, subdomain=grid.subdomains['sd_right'])

        op = Operator([eqn1, eqn2, eqn3])
        op.apply()

        expected = np.array([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 2., 2., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 2., 2., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 3., 3.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 3., 3.]], dtype=np.float32)

        assert ((np.array(v.data[:]) == expected).all())

    @pytest.mark.parallel(mode=4)
    def test_function_subdomain_mpi(self):
        """
        Tests with the Function is allocated with the correct size, and if it works with
        mpi.
        Expected output:

              rank0         rank1
            0 0 0 0 0     0 0 0 0 0
            0 0 0 0 0     0 0 0 0 0
            0 0 0 0 0     0 0 0 0 0
            0 0 0 2 2     2 2 0 0 0
            0 0 0 2 2     2 2 0 0 0

              rank2         rank3
            0 0 0 2 2     2 2 0 0 0
            0 0 0 2 2     2 2 0 0 0
            0 0 0 0 0     0 0 0 0 0
            0 0 0 0 0     0 0 0 0 0
            0 0 0 0 0     0 0 0 0 0
        """

        class SubDomain_Middle(SubDomain):
            name = 'sd_middle'

            def define(self, dimensions):
                x, y = dimensions
                # Create a 3x3 subdomain in the middle of the grid.
                return {x: ('middle', 3, 3), y: ('middle', 3, 3)}

        sd_middle = SubDomain_Middle()

        grid = Grid((10, 10), subdomains=(sd_middle))
        x, y = grid.dimensions

        u = Function(name='u', grid=grid, subdomain=sd_middle)
        v = Function(name='v', grid=grid)

        assert u.shape == (4, 4)

        eqn = Eq(v, u + 2, subdomain=grid.subdomains['sd_middle'])

        op = Operator(eqn)
        op.apply()

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            expected = np.array([[0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 2., 2.,],
                                 [0., 0., 0., 2., 2.,]], dtype=np.float)
            assert ((np.array(v.data[:]) == expected).all())

        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            expected = np.array([[0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,],
                                 [2., 2., 0., 0., 0.,],
                                 [2., 2., 0., 0., 0.,]], dtype=np.float)
            assert ((np.array(v.data[:]) == expected).all())

        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            expected = np.array([[0., 0., 0., 2., 2.,],
                                 [0., 0., 0., 2., 2.,],
                                 [0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,]], dtype=np.float)
            assert ((np.array(v.data[:]) == expected).all())

        elif RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            expected = np.array([[2., 2., 0., 0., 0.,],
                                 [2., 2., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,],
                                 [0., 0., 0., 0., 0.,]], dtype=np.float)
            assert ((np.array(v.data[:]) == expected).all())
