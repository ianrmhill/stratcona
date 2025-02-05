
import numpyro as npyro
import numpyro.distributions as dists
from numpyro.handlers import seed, trace, condition

import jax.numpy as jnp
import jax.random as rand
import jax

from functools import partial
import timeit

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona


def main():

    key = rand.key(363)
    k1, k2 = rand.split(key)

    def basic_model(n_dev):
        a = npyro.sample('a', dists.Normal(0, 1))
        b = npyro.sample('b', dists.Normal(0, 1))
        with npyro.plate('dev', n_dev):
            npyro.sample('y', dists.Normal(a, b))

    def basic_sample(k, n_dev):
        return seed(basic_model, k)(n_dev)

    to_time = partial(basic_sample, k1, 50)
    best_perf = min(timeit.Timer(to_time).repeat(repeat=100, number=100))
    print(f'Best unjitted: {best_perf}')

    # Now jit compile it
    j_sample = jax.jit(basic_sample, static_argnames='n_dev')
    # Run it once with the correct static args to compile before timing
    j_sample(k1, 500)

    to_time = partial(j_sample, k1, 500)
    best_perf = min(timeit.Timer(to_time).repeat(repeat=100, number=100))
    print(f'Best jitted: {best_perf}')


if __name__ == '__main__':
    main()
