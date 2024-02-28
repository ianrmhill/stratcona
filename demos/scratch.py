import pytensor
import pytensor.tensor as pt
from pytensor.scan.utils import until

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stratcona.engine.minimization import brent_minimize_std

# Our default machine precision / machine epsilon is based on a 32-bit computer IEEE floating point representation,
# this avoids extra work for little accuracy gain on 32-bit processors.
# The value can be increased if doubles are certain to be used, or reduced for use cases not requiring high accuracy
MACH_32BIT_EPS = 2 ** -24

# These values are used when segmenting golden search intervals to maximize rate of convergence
GOLDEN_RATIO = (1 + (5**0.5)) / 2
GF1 = GOLDEN_RATIO - 1 # approx. 0.618
GF2 = 2 - GOLDEN_RATIO # approx. 0.382 (in other words, 1 - GF1)


def sample_f(x, placement):
    y = (x-placement)**2
    return y


F = sample_f


def minimize_golden(func, extra_args, bounds, precision=1e-2, maxiter=100):
    """
    Implementation of golden section search for interim while I get Brent minimization working.
    """
    # Initialize the bounds a and b within which to find the function minimum
    b_l = pytensor.printing.Print('b_l')(pt.extra_ops.repeat(bounds[0], 3))
    b_u = pytensor.printing.Print('b_u')(pt.extra_ops.repeat(bounds[1], 3))

    def golden_section_search_loop(l, u, eps):
        # Compute the interior points and function evaluations
        c = l + (GF2 * (u - l))
        d = u - (GF2 * (u - l))
        fc = func(c, **extra_args)
        fd = func(d, **extra_args)

        # Compare the function evaluated at the two interior points 'u' and 'x' to determine how to reduce the interval
        l = pt.where(fd >= fc, l, c)
        u = pt.where(fd >= fc, d, u)

        return (l, u), until(pt.all((u - l) < eps))

    values, _ = pytensor.scan(golden_section_search_loop, outputs_info=(b_l, b_u), non_sequences=[precision], n_steps=maxiter)

    return values[0][-1], values[1][-1], (values[0][-1] + values[1][-1]) / 2


def minimize(func, extra_args, bounds, precision=1e-5, mach_eps=MACH_32BIT_EPS, maxiter=20):
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
    # Initialize the bounds a and b within which to find the function minimum
    num_problems = len(next(iter(extra_args.values())))
    a = pt.extra_ops.repeat(bounds[0], num_problems)
    b = pt.extra_ops.repeat(bounds[1], num_problems)
    # Initialize the parabolic interpolation points
    v = w = x = a + (GF2 * (b - a))
    as_kwargs = {}
    for i, arg in enumerate(f_args_order):
        as_kwargs[arg] = f_args[i]
    fv = fw = fx = func(x, **as_kwargs)
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

        use_para = pytensor.printing.Print('interpolate')( use_para & pt.neq(q, 0.0) )
        p = pt.where(use_para, (xdiff2 * qsub2) - (xdiff1 * qsub1), 0.0)
        p = pt.where(pt.gt(q, 0.0), -p, p)
        q = pt.abs(q)

        further_boundary = pt.where(pt.ge(x, m), a, b)
        step = pt.where(use_para, p / q, GF2 * (further_boundary - x))
        u_tent = pytensor.printing.Print('u_prop')( x + step )

        out_of_bounds = pt.lt(b, u_tent) | pt.gt(a, u_tent) | pt.gt(pt.abs(step), (0.5 * pt.abs(e)))
        change_to_gold = use_para & out_of_bounds
        # Determine the golden search step for any problems where parabolic search just failed the final two checks
        step = pt.where(change_to_gold, GF2 * (further_boundary - x), step)
        # Ensure all new points 'u' will be at least 'tol' away from 'x' and at least 2 'tol' from the interval bounds
        step = pt.where(pt.lt(u_tent - a, tol2) | pt.lt(b - u_tent, tol2), pt.where(pt.lt(x, m), tol, -tol), step)
        step = pt.where(pt.lt(pt.abs(step), tol), pt.where(pt.gt(step, 0), tol, -tol), step)
        # Compute the new points
        u = pytensor.printing.Print('u')(x + step)
        # CRUCIAL NOTE: If the f_kwargs are not provided as explicit inputs to scan then they are treated as static,
        # thus their value is fixed at pytensor function compilation. The function 'func' is okay to treat as static
        # since the function definition won't change once the model is set up (also scan can't handle a function as an
        # argument).
        kwargs = {}
        for i, arg in enumerate(f_args_order):
            kwargs[arg] = f_args[i]
        fu = func(u, **kwargs)

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

        x, fx = pytensor.printing.Print('x')( pt.where(is_less, u, x) ), pt.where(is_less, fu, fx)

        # Check whether our estimates are all good enough
        m = (b + a) / 2.0
        tol = (rel_tol * pt.abs(x)) + abs_tol
        tol2 = tol * 2.0
        return (a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2), until(pt.all(pt.lt(pt.abs(x - m), tol2 - (b - m))))

    values, _ = pytensor.scan(brent_minimization_loop, outputs_info=(a, b, v, w, x, fv, fw, fx, d, e, m, tol, tol2),
                              non_sequences=[*extra_args.values()], n_steps=maxiter)

    # Return the value of x from the final iteration and the function evaluated at x for verification
    return values[4][-1], values[7][-1]


#def test_func():
#    t = {'t': pt.zeros((3, 4, 2))}
#    a = pytensor.printing.Print('a')( pt.fill(next(iter(t.values())), 3) )
#    return a
#
#dim1, dim2 = pt.scalar(), pt.scalar()
#func_test = test_func()
#compiled = pytensor.function([], func_test)
#compiled()

lower_bound, upper_bound = pt.scalar(), pt.scalar()
func_min = minimize(F, {'placement': [1, 6, -3, -6]}, (lower_bound, upper_bound))
compiled = pytensor.function([lower_bound, upper_bound], func_min)

print(compiled(-8, 10))

x, fx = brent_minimize_std(F, {'placement': -6}, (-8, 10))
print(f"Min x: {x}, f(x): {fx}")

