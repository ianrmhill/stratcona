
import numpyro as npyro
import numpyro.distributions as dists
from numpyro.handlers import seed, trace, condition

import jax.numpy as jnp
import jax.random as rand
import jax

from dataclasses import dataclass
from functools import partial
import timeit

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona


@dataclass(frozen=True)
class ExpDims:
    name: str
    lot: int
    chp: int

    def __init__(self, name, lot, chp):
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'lot', lot)
        object.__setattr__(self, 'chp', chp)

    def __members(self):
        return self.name, self.lot, self.chp

    def __hash__(self):
        return hash(self.__members())

    def __eq__(self, other):
        return type(self) == type(other) and self.__members() == other.__members()


@dataclass(frozen=True)
class TestDims():
    name: str
    exps: tuple[ExpDims, ...]

    def __init__(self, name, config):
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'exps', tuple([ExpDims(name=exp, **config[exp]) for exp in config]))

    def __members(self):
        return self.name, *self.exps

    def __hash__(self):
        return hash(self.__members)

    def __eq__(self, other):
        return type(self) == type(other) and self.__members() == other.__members()


def classes():
    e1 = ExpDims('f', 2, 2)
    e2 = ExpDims('g', 3, 2)
    e3 = ExpDims('g', 3, 2)
    print(e1 == e2)
    print(e2 == e3)
    t1 = TestDims('mytest', {'f': {'lot': 2, 'chp': 2}, 'g': {'lot': 3, 'chp': 2}})
    t3 = TestDims('mytest', {'g': {'lot': 3, 'chp': 2}, 'f': {'lot': 2, 'chp': 2}})
    t2 = TestDims('2test', {'f': {'lot': 2, 'chp': 2}})
    print(hash(t1))
    print(hash(t3))
    print(t1 == t2)
    print(t1 == t1)


def main():
    key = rand.key(363)
    k1, k2 = rand.split(key)

    tf = dists.transforms.SoftplusTransform()

    def basic_model(dims, priors, conds):
        a = npyro.sample('a', dists.Normal(**priors['a']))
        b = npyro.sample('b', dists.TransformedDistribution(dists.Normal(**priors['b']), tf))
        for exp in dims.exps:
            with npyro.plate('chp', exp.chp):
                pre_y = npyro.deterministic('pre_y', a * conds[exp.name]['temp'])
                npyro.sample('y', dists.Normal(pre_y, b))

    def basic_sample(k, dims, priors, conds):
        return trace(seed(basic_model, k)).get_trace(dims, priors, conds)['y']['value']

    dims = TestDims('t1', {'e': {'lot': 1, 'chp': 50}})
    pri = {'a': {'loc': 1, 'scale': 0.5}, 'b': {'loc': 0.5, 'scale': 0.2}}
    c = {'e': {'temp': 3}}
    to_time = partial(basic_sample, k1, dims, pri, c)
    best_perf = min(timeit.Timer(to_time).repeat(repeat=100, number=100))
    print(f'Best unjitted: {best_perf}')

    # Now jit compile it
    j_sample = jax.jit(basic_sample, static_argnames='dims')
    # Run it once with the correct static args to compile before timing
    j_sample(k1, dims, pri, c)
    d2 = TestDims('t2', {'e': {'lot': 1, 'chp': 5}})
    print(j_sample(k1, d2, pri, c))

    to_time = partial(j_sample, k1, dims, pri, c)
    best_perf = min(timeit.Timer(to_time).repeat(repeat=100, number=100))
    print(f'Best jitted: {best_perf}')


if __name__ == '__main__':
    main()
