import pytensor as pt
from pytensor.scan.utils import until

GOLDEN_RATIO = (1 + (5**0.5)) / 2
GF1 = GOLDEN_RATIO - 1 # approx. 0.618
GF2 = 2 - GOLDEN_RATIO # approx. 0.382 (in other words, 1 - GF1)


def sample_f(x, placement):
    y = (x-placement)**2
    return y


F = sample_f


def minimize(func, extra_args, bounds, precision=1e-2, maxiter=100):
    """
    Implementation of golden section search for interim while I get Brent minimization working.
    """
    # Initialize the bounds a and b within which to find the function minimum
    b_l = pt.printing.Print('b_l')(pt.tensor.extra_ops.repeat(bounds[0], 3))
    b_u = pt.printing.Print('b_u')(pt.tensor.extra_ops.repeat(bounds[1], 3))

    def golden_section_search_loop(l, u, eps):
        # Compute the interior points and function evaluations
        c = l + (GF2 * (u - l))
        d = u - (GF2 * (u - l))
        fc = func(c, **extra_args)
        fd = func(d, **extra_args)

        # Compare the function evaluated at the two interior points 'u' and 'x' to determine how to reduce the interval
        l = pt.tensor.where(fd >= fc, l, c)
        u = pt.tensor.where(fd >= fc, d, u)

        return (l, u), until(pt.tensor.all((u - l) < eps))

    values, _ = pt.scan(golden_section_search_loop, outputs_info=(b_l, b_u), non_sequences=[precision], n_steps=maxiter)

    return values[0][-1], values[1][-1], (values[0][-1] + values[1][-1]) / 2


lower_bound, upper_bound = pt.tensor.scalar(), pt.tensor.scalar()
func_min = minimize(F, {'placement': [2, 5, -3]}, (lower_bound, upper_bound), 0.01, 50)
compiled = pt.function([lower_bound, upper_bound], func_min)

print(compiled(-8, 10))
