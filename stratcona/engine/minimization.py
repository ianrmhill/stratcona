# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import pytensor as pt

GOLDEN_RATIO = (1 + (5**0.5)) / 2
GF1 = GOLDEN_RATIO - 1 # approx. 0.618
GF2 = 2 - GOLDEN_RATIO # approx. 0.382 (in other words, 1 - GF1)


def minimize(f, extra_args, bounds, precision=1e-3, maxiter=100):
    """
    Implementation of golden section search for interim while I get Brent minimization working.
    """
    # Initialize the bounds a and b within which to find the function minimum
    b_l, b_u = bounds[0], bounds[1]
    if b_l >= b_u:
        raise Exception(f"Bounds not valid, upper bound {b_u} must be greater than the lower bound {b_l}.")

    def golden_section_search_loop(b_l, b_u):
        # Compute the interior points and function evaluations
        c, d = b_l + (GF2 * (b_u - b_l)), b_u - (GF2 * (b_u - b_l))
        fc, fd = f(c, extra_args), f(c, extra_args)

        # Compare the function evaluated at the two interior points 'u' and 'x' to determine how to reduce the interval
        b_l = pt.tensor.where(pt.tensor.ge(fd, fc), b_l, c)
        b_u = pt.tensor.where(pt.tensor.ge(fd, fc), d, b_u)

    pt.scan(golden_section_search_loop)


    curr_i = 0
    while True:
        # Compute the interior points and function evaluations
        c, d = b_l + (GF2 * (b_u - b_l)), b_u - (GF2 * (b_u - b_l))
        fc, fd = f(c, extra_args), f(c, extra_args)

        # Compare the function evaluated at the two interior points 'u' and 'x' to determine how to reduce the interval
        b_l = pt.tensor.where(pt.tensor.ge(fd, fc), b_l, c)
        b_u = pt.tensor.where(pt.tensor.ge(fd, fc), d, b_u)

        curr_i += 1
        # Check for termination, which occurs if the bounded zone is less than the precision or we've hit max iterations
        if pt.tensor.all(pt.tensor.lt(b_u - b_l, precision)) or curr_i > maxiter:
            if curr_i > maxiter:
                print('Warning: hit max iterations of minimization method, calculated times may not be accurate')
            # Return the midpoint of the bounded region, which is our final estimate once the region is very small
            print(f"Iterations: {curr_i}")
            return (b_u + b_l) / 2


def minimize_inprogress(f, extra_args, bounds, precision=1e-3, maxiter=10):
    """
    Implementation of Brent's minimization method compatible with pytensor and TODO: hopefully vectorization.

    Variable naming and organization closely follows the explanation by Oscar Veliz found at:
    https://www.youtube.com/watch?v=BQm7uTYC0sg&t=1s

    Parameters
    ----------
    f
    extra_args
    bounds
    precision
    maxiter

    Returns
    -------

    """
    # Initialize the bounds a and b within which to find the function minimum
    b_l, b_u = bounds[0], bounds[1]
    if b_l >= b_u:
        raise Exception(f"Bounds not valid, upper bound {b_u} must be greater than the lower bound {b_l}.")
    # Think of v, w, x as x_(n-2), x_(n-1), and x_n, each parabolic interpolation iteration finds x_(n+1)
    v = w = x = b_l + (GF2 * (b_u - b_l))
    fv = fw = fx = f(time=x, **extra_args)

    curr_i = 0
    while True:
        curr_i += 1
        # Compute the midpoint of the bounded region, which will be our final estimate once the region is very small
        m = (b_u + b_l) / 2
        # Check for termination, which occurs if the bounded zone is less than the precision or we've hit max iterations
        if b_u - b_l < precision or curr_i > maxiter:
            if curr_i > maxiter:
                print('Warning: hit max iterations of minimization method, calculated times may not be accurate')
            return m

        # For next iteration, determine whether to run successive parabolic interpolation or golden section search step
        # We use interpolation whenever it will behave nicely, thus check if it will first
        # The checks are
        # 1) q is not 0, which would make x_step infinite
        # 2) the new value 'u' is within the current bounded region
        # 3) x_step is less than half the magnitude of x_step_prev (in other words, we should be converging quickly)
        # 4) a min step size check to ensure we're still moving and not stationary (TODO: check this)
        xdiff1, xdiff2 = w - x, v - x
        x_step_prev = xdiff1
        # Compute the two sub-components of q
        qsub1, qsub2 = xdiff1 * (fx - fv), xdiff2 * (fw - fx)
        q = 2 * (qsub1 + qsub2)
        if q != 0:
            p = (xdiff1 * qsub1) + (xdiff2 * qsub2)
            x_step = p / q
            u = x + x_step
            use_parabolic = q != 0 and b_l < u < b_u and abs(x_step) < (0.5 * abs(x_step_prev)) and (x_step > 'tolerance')

        # If we reach this point, we need to perform a golden section search instead
        # Compute the second point for comparison within the search interval using the golden ratio
        u = b_l + (GF1 * (x - b_l)) if x >= m else b_u + (GF1 * (x - b_u))
        fu = f(time=u, **extra_args)
        # Compare the function evaluated at the two interior points 'u' and 'x' to determine how to reduce the interval
        if u < x:
            b_l_next = u if fu >= fx else b_l
            b_u_next = b_u if fu >= fx else x
        else:
            b_l_next = b_l if fu >= fx else x
            b_u_next = u if fu >= fx else b_u

        # Now update the various points in preparation for the next iteration
        v, fv = w, fw
        w, fw = x, fx
        x, fx = u, fu
        b_l = b_l_next
        b_u = b_u_next
