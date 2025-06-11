# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from functools import partial
import jax
import jax.numpy as jnp

import pytensor
import pytensor.tensor as pt
from pytensor.scan.utils import until

# Our default machine precision / machine epsilon is based on a 32-bit computer IEEE floating point representation,
# this avoids extra work for little accuracy gain on 32-bit processors.
# The value can be increased if doubles are certain to be used, or reduced for use cases not requiring high accuracy
MACH_32BIT_EPS = 2 ** -24

# These values are used when segmenting golden search intervals to maximize rate of convergence
GOLDEN_RATIO = (1 + (5**0.5)) / 2
GF1 = GOLDEN_RATIO - 1 # approx. 0.618
GF2 = 2 - GOLDEN_RATIO # approx. 0.382 (in other words, 1 - GF1)


@partial(jax.jit, static_argnames=['func', 'maxiter', 'log_gold'])
def minimize_jax(func, extra_args: dict, bounds, precision=1e-5, mach_eps=MACH_32BIT_EPS, maxiter=20, log_gold=False):
    f_args_order = extra_args.keys()
    f_args = list(extra_args.values())
    # We divide by three so that abs_tol directly represents the furthest that the estimated minimum can be from the
    # true minimum
    abs_tol = precision / 3.0
    rel_tol = mach_eps ** 0.5
    # Need to determine the shape of any non-singular so that the number of parallel solutions is known
    shape_to_match = f_args[0]
    for arg in f_args:
        if hasattr(arg, 'shape') and arg.shape != ():
            shape_to_match = arg
    # Initialize the bounds a and b within which to find the function minimum, shape determined by first function arg
    a = jnp.full_like(shape_to_match, jnp.log10(bounds[0]) if log_gold else bounds[0], dtype=jnp.float32)
    b = jnp.full_like(shape_to_match, jnp.log10(bounds[1]) if log_gold else bounds[1], dtype=jnp.float32)
    # Initialize the parabolic interpolation points
    v = w = x = a + (GF2 * (b - a))
    as_kwargs = {}
    for i, arg in enumerate(f_args_order):
        as_kwargs[arg] = f_args[i]
    fv = fw = fx = func(10 ** x if log_gold else x, **as_kwargs)
    # 'e' represents the step size between the old and new minimum estimate from two iterations ago. This bookkeeping
    # is used to force golden section searches if parabolic interpolation is not proceeding faster than a bisection
    # search method. Not strictly necessary for guaranteed convergence, but adds a guarantee that the algorithm is never
    # much slower than a raw golden section search algorithm. Brent found that using 'e' from two iterations ago worked
    # better than using 'd' from one iteration ago, no hard mathematical justification for the choice.
    e = d = jnp.zeros_like(a, dtype=jnp.float32)

    # Compute the midpoint of the bounded region, which will be our final estimate once the region is very small
    m = (b + a) / 2.0
    # The function will never be evaluated at two points closer together than the tolerance
    tol = (rel_tol * jnp.abs(x)) + abs_tol
    # Value used multiple times, do the multiplication only once here to save time
    tol2 = tol * 2.0

    i = 0
    init_vals = (a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2, i)
    def stop(vals):
        a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2, i = vals
        return jnp.less(i, maxiter) & ~jnp.all(jnp.less(jnp.abs(x - m), tol2 - (b - m)))

    def brent_iter(vals):
        a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2, i = vals
        i += 1
        # Start by identifying which problems are candidates for a parabolic interpolation step
        use_para = jnp.greater(jnp.abs(e), tol)
        xdiff1 = jnp.where(use_para, x - w, 0.0)
        xdiff2 = jnp.where(use_para, x - v, 0.0)
        # Currently doing for all as the diffs will be zeros for golden search problems, adds little computation
        qsub1 = xdiff1 * (fx - fv)
        qsub2 = xdiff2 * (fx - fw)
        q = 2 * (qsub2 - qsub1)

        use_para = use_para & jnp.not_equal(q, 0.0)
        p = jnp.where(use_para, (xdiff2 * qsub2) - (xdiff1 * qsub1), 0.0)
        p = jnp.where(jnp.greater(q, 0.0), -p, p)
        q = jnp.abs(q)

        further_boundary = jnp.where(jnp.greater_equal(x, m), a, b)
        step = jnp.where(use_para, p / q, GF2 * (further_boundary - x))
        u_tent = x + step

        change_to_gold = use_para & (jnp.less(b, u_tent) | jnp.greater(a, u_tent) | jnp.greater(jnp.abs(step), (0.5 * jnp.abs(e))))
        # Determine the golden search step for any problems where parabolic search just failed the final two checks
        step = jnp.where(change_to_gold, GF2 * (further_boundary - x), step)
        u_tent = jnp.where(change_to_gold, x + step, u_tent)
        # Ensure all new points 'u' will be at least 'tol' away from 'x' and at least 2 'tol' from the interval bounds
        step = jnp.where(jnp.less(u_tent - a, tol2) | jnp.less(b - u_tent, tol2), jnp.where(jnp.less(x, m), tol, -tol), step)
        step = jnp.where(jnp.less(jnp.abs(step), tol), jnp.where(jnp.greater(step, 0), tol, -tol), step)
        # Compute the new points
        u = x + step
        fu = func(10 ** u if log_gold else u, **as_kwargs)

        # Now to update all the variables
        e = d
        d = step
        is_less = jnp.less_equal(fu, fx)

        a = jnp.where(is_less & jnp.greater_equal(u, x), x, jnp.where(~is_less & jnp.less(u, x), u, a))
        b = jnp.where(is_less & jnp.less(u, x), x, jnp.where(~is_less & jnp.greater_equal(u, x), u, b))

        v_cond_1, v_cond_2 = is_less | jnp.less_equal(fu, fw) | jnp.equal(w, x), jnp.less_equal(fu, fv) | jnp.equal(v, w) | jnp.equal(v, x)
        v, fv = jnp.where(v_cond_1, w, jnp.where(v_cond_2, u, v)), jnp.where(v_cond_1, fw, jnp.where(v_cond_2, fu, fv))

        w_cond = jnp.less_equal(fu, fw) | jnp.equal(w, x)
        w, fw = jnp.where(is_less, x, jnp.where(w_cond, u, w)), jnp.where(is_less, fx, jnp.where(w_cond, fu, fw))

        x, fx = jnp.where(is_less, u, x), jnp.where(is_less, fu, fx)

        # Check whether our estimates are all good enough
        m = (b + a) / 2.0
        tol = (rel_tol * jnp.abs(x)) + abs_tol
        tol2 = tol * 2.0

        return a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2, i

    res = jax.lax.while_loop(stop, brent_iter, init_vals)
    x = res[4]

    return 10 ** x if log_gold else x


def minimize(func, extra_args: dict, bounds, precision=1e-5, mach_eps=MACH_32BIT_EPS, maxiter=20, log_gold=False):
    """
    Implementation of Brent minimization with vectorization and pytensor compatibility (i.e., can be compiled into a
    graphical model).
    """
    f_args_order = extra_args.keys()
    f_args = list(extra_args.values())
    # We divide by three so that abs_tol directly represents the furthest that the estimated minimum can be from the
    # true minimum
    abs_tol = precision / 3.0
    rel_tol = mach_eps ** 0.5
    # Initialize the bounds a and b within which to find the function minimum, shape determined by first function arg
    shape_to_match = f_args[0]
    a = pt.fill(shape_to_match, pt.log10(bounds[0])) if log_gold else pt.fill(shape_to_match, bounds[0])
    b = pt.fill(shape_to_match, pt.log10(bounds[1])) if log_gold else pt.fill(shape_to_match, bounds[1])
    # Initialize the parabolic interpolation points
    v = w = x = a + (GF2 * (b - a))
    as_kwargs = {}
    for i, arg in enumerate(f_args_order):
        as_kwargs[arg] = f_args[i]
    fv = fw = fx = func(10 ** x, **as_kwargs) if log_gold else func(x, **as_kwargs)
    # 'e' represents the step size between the old and new minimum estimate from two iterations ago. This bookkeeping
    # is used to force golden section searches if parabolic interpolation is not proceeding faster than a bisection
    # search method. Not strictly necessary for guaranteed convergence, but adds a guarantee that the algorithm is never
    # much slower than a raw golden section search algorithm. Brent found that using 'e' from two iterations ago worked
    # better than using 'd' from one iteration ago, no hard mathematical justification for the choice.
    e = d = pt.zeros_like(a)

    # Compute the midpoint of the bounded region, which will be our final estimate once the region is very small
    m = (b + a) / 2.0
    # The function will never be evaluated at two points closer together than the tolerance
    tol = (rel_tol * pt.abs(x)) + abs_tol
    # Value used multiple times, do the multiplication only once here to save time
    tol2 = tol * 2.0

    def brent_minimization_loop(a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2, *f_args):
        # Start by identifying which problems are candidates for a parabolic interpolation step
        use_para = pt.gt(pt.abs(e), tol)
        xdiff1 = pt.where(use_para, x - w, 0.0)
        xdiff2 = pt.where(use_para, x - v, 0.0)
        # Currently doing for all as the diffs will be zeros for golden search problems, adds little computation
        qsub1 = xdiff1 * (fx - fv)
        qsub2 = xdiff2 * (fx - fw)
        q = 2 * (qsub2 - qsub1)

        use_para = use_para & pt.neq(q, 0.0)
        p = pt.where(use_para, (xdiff2 * qsub2) - (xdiff1 * qsub1), 0.0)
        p = pt.where(pt.gt(q, 0.0), -p, p)
        q = pt.abs(q)

        further_boundary = pt.where(pt.ge(x, m), a, b)
        step = pt.where(use_para, p / q, GF2 * (further_boundary - x))
        u_tent = x + step

        change_to_gold = use_para & (pt.lt(b, u_tent) | pt.gt(a, u_tent) | pt.gt(pt.abs(step), (0.5 * pt.abs(e))))
        # Determine the golden search step for any problems where parabolic search just failed the final two checks
        step = pt.where(change_to_gold, GF2 * (further_boundary - x), step)
        u_tent = pt.where(change_to_gold, x + step, u_tent)
        # Ensure all new points 'u' will be at least 'tol' away from 'x' and at least 2 'tol' from the interval bounds
        step = pt.where(pt.lt(u_tent - a, tol2) | pt.lt(b - u_tent, tol2), pt.where(pt.lt(x, m), tol, -tol), step)
        step = pt.where(pt.lt(pt.abs(step), tol), pt.where(pt.gt(step, 0), tol, -tol), step)
        # Compute the new points
        u = x + step
        # CRUCIAL NOTE: If the f_kwargs are not provided as explicit inputs to scan then they are treated as static,
        # thus their value is fixed at pytensor function compilation. The function 'func' is okay to treat as static
        # since the function definition won't change once the model is set up (also scan can't handle a function as an
        # argument).
        fn_kwargs = {}
        for i, arg in enumerate(f_args_order):
            fn_kwargs[arg] = f_args[i]
        fu = func(10 ** u, **fn_kwargs) if log_gold else func(u, **fn_kwargs)

        # Now to update all the variables
        e = d
        d = step
        is_less = pt.le(fu, fx)

        a = pt.where(is_less & pt.ge(u, x), x, pt.where(~is_less & pt.lt(u, x), u, a))
        b = pt.where(is_less & pt.lt(u, x), x, pt.where(~is_less & pt.ge(u, x), u, b))

        v_cond_1, v_cond_2 = is_less | pt.le(fu, fw) | pt.eq(w, x), pt.le(fu, fv) | pt.eq(v, w) | pt.eq(v, x)
        v, fv = pt.where(v_cond_1, w, pt.where(v_cond_2, u, v)), pt.where(v_cond_1, fw, pt.where(v_cond_2, fu, fv))

        w_cond = pt.le(fu, fw) | pt.eq(w, x)
        w, fw = pt.where(is_less, x, pt.where(w_cond, u, w)), pt.where(is_less, fx, pt.where(w_cond, fu, fw))

        x, fx = pt.where(is_less, u, x), pt.where(is_less, fu, fx)

        # Check whether our estimates are all good enough
        m = (b + a) / 2.0
        tol = (rel_tol * pt.abs(x)) + abs_tol
        tol2 = tol * 2.0
        return (a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2), until(pt.all(pt.lt(pt.abs(x - m), tol2 - (b - m))))

    values, _ = pytensor.scan(brent_minimization_loop, outputs_info=(a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2),
                              non_sequences=[*f_args], n_steps=maxiter, name='time-solver')

    # TODO: Figure out how to print a warning conditionally in the compiled version if maxiter is hit
    # Return the value of x from the final iteration and the function evaluated at x for verification
    return 10 ** values[4][-1] if log_gold else values[4][-1]


def brent_minimize_std(f, extra_args, bounds, maxiter=10, rel_tol=MACH_32BIT_EPS**0.5, abs_tol=1e-5):
    """
    Implementation of Brent's minimization method in straightforward Python, helps for understanding the Pytensor
    vectorized version.

    This tries to follow the explanation in Algorithms for Minimization Without Derivatives by Richard P. Brent, 1973.

    A decent explanation of the algorithm is provided by Oscar Veliz, found at:
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
    # We divide by three so that abs_tol directly represents the furthest that the estimated minimum can be from the
    # true minimum
    max_error = abs_tol / 3.0
    # 'e' represents the step size between the old and new minimum estimate from two iterations ago. This bookkeeping
    # is used to force golden section searches if parabolic interpolation is not proceeding faster than a bisection
    # search method. Not strictly necessary for guaranteed convergence, but adds a guarantee that the algorithm is never
    # much slower than a raw golden section search algorithm.
    e = d = 0.0
    # Initialize the bounds a and b within which to find the function minimum
    b_l, b_u = bounds[0], bounds[1]
    if b_l >= b_u:
        raise Exception(f"Bounds not valid, upper bound {b_u} must be greater than the lower bound {b_l}.")
    # Think of v, w, x as x_(n-2), x_(n-1), and x_n, each parabolic interpolation iteration finds x_(n+1)
    v = w = x = b_l + (GF2 * (b_u - b_l))
    fv = fw = fx = f(x, **extra_args)

    curr_iter = 0
    while True:
        curr_iter += 1
        golden = False
        # Compute the midpoint of the bounded region, which will be our final estimate once the region is very small
        m = (b_u + b_l) / 2
        # The function will never be evaluated at two points closer together than the tolerance, though this value is a
        # combination of both an absolute tolerance t and a relative min spacing based on the magnitude of x.
        tol = rel_tol * abs(x) + max_error
        # Value used multiple times, do the multiplication only once here to save time
        tol_2 = tol * 2.0

        # Check for termination, which occurs once the current minimum x is guaranteed to be less than 2 tolerances from
        # either edge of the bounded interval (or if we run out of iterations)
        if abs(x - m) < tol_2 - (b_u - m) or curr_iter > maxiter:
            if curr_iter > maxiter:
                print('Warning: Hit max iterations of minimization method, estimates may not have desired accuracy.')
            return x, fx

        # Perform the setup for parabolic interpolation, there are 4 checks to see if golden search should be used instead
        # The first: 1) If the step size from two iterations ago was already smaller than the tolerance, we will hit
        # convergence with a golden search step, so skip work of interpolation setup
        step = 0.0
        if abs(e) > tol:
            # Calculate q, doing some intermediate steps to save work computing p
            xdiff1, xdiff2 = x - w, x - v
            qsub1, qsub2 = xdiff1 * (fx - fv), xdiff2 * (fx - fw)
            q = 2 * (qsub2 - qsub1)
            # Check that: 2) q is not 0, which would make p/q infinite
            if q != 0:
                p = (xdiff2 * qsub2) - (xdiff1 * qsub1)
                # Check signs are correct for the interpolation step
                p = -p if q > 0 else p
                q = abs(q)
                # Determine the step distance and next estimate for the minimum, 'u'
                step = p / q
                u_proposal = x + step
                # Check that: 3) u is within the bounded interval, and that
                # 4) the step size is less than half the step size from two iters ago, i.e., we are converging quickly
                if b_u > u_proposal > b_l and abs(step) < (0.5 * abs(e)):
                    # Now ensure u isn't too close to x, b_u, or b_l to be evaluated, in essence guarantees that the
                    # algorithm will ignore dips (local minimums) in the function less than the width of the tolerance
                    if u_proposal - b_l < tol_2 or b_u - u_proposal < tol_2:
                        # Direction of small step chosen to be towards midpoint of bounded interval
                        step = tol if x < m else -tol
                else:
                    golden = True
            else:
                golden = True
        else:
            golden = True

        # If we reach this point, we need to perform a golden section search instead
        # Compute the second point for comparison within the search interval using the golden ratio
        if golden:
            # Determine the step required to get to the new point u to compare against x
            step = GF2 * ((b_l if x >= m else b_u) - x)

        # Ensure step isn't too small, algorithm guarantees f will not be evaluated at any two points closer than tol
        if abs(step) < tol:
            step = tol if step > 0 else -tol

        # Now compute the new point for certain
        u = x + step
        fu = f(u, **extra_args)

        # Now update the various points in preparation for the next iteration
        e = d
        d = step
        if fu <= fx:
            # fu <= fx is always true if we did parabolic interpolation, and requires no reordering of variables if true
            # in the golden section search case
            b_l = b_l if u < x else x # Note: interval bounds always update so that max(fu, fx) is now a boundary
            b_u = b_u if u >= x else x
            v, w, x, fv, fw, fx = w, x, u, fw, fx, fu
        else:
            # If fu > fx in a golden section search step then we need to reorder our points to fix the sequencing
            b_l = b_l if u >= x else u
            b_u = b_u if u < x else u
            # If fu is our new second least point or if we are near the start of the algorithm and can't yet do interpolation steps
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            # If fu is our new third least point or if we can't do interpolation because we have overlapping points
            elif fu <= fv or v == w or v == x:
                v, fv = u, fu
            # If none of the above, then fu > any of fv, fw, fx and so u is discarded (u is now a boundary, not a point)
            else:
                pass
