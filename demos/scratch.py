import pytensor
import pytensor.tensor as pt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stratcona.engine.minimization import minimize, brent_minimize_std


def sample_f(x, placement):
    y = (x-placement)**2
    return y


F = sample_f


lower_bound, upper_bound = pt.scalar(), pt.scalar()
func_min = minimize(F, {'placement': [2, 6, 354, 87]}, (lower_bound, upper_bound), log_gold=True)
compiled = pytensor.function([lower_bound, upper_bound], func_min)

print(compiled(1, 1e4))

x, fx = brent_minimize_std(F, {'placement': -6}, (-8, 10))
print(f"Min x: {x}, f(x): {fx}")

