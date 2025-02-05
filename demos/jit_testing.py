
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
from stratcona.modelling.relmodel import TestDef, ExpDims


def main():
    key = rand.key(363)
    k1, k2 = rand.split(key)

    ####################################################
    # Define an NBTI empirical model within the SPM framework for inference
    ####################################################
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb = stratcona.SPMBuilder(mdl_name='bti-empirical')
    mb.add_params(k=8.617e-5, zero=0.0, meas_var=2, n_nom=2, alpha=3.5, n=2)

    # Initial parameters are simulating some data to then learn
    mb.add_hyperlatent('a0_nom', dists.Normal, {'loc': 5, 'scale': 0.01})
    mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 6, 'scale': 0.01})

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('a0_dev', dists.Normal, {'loc': 6, 'scale': 0.01}, transform=var_tf)
    mb.add_hyperlatent('a0_chp', dists.Normal, {'loc': 2, 'scale': 0.01}, transform=var_tf)
    mb.add_hyperlatent('a0_lot', dists.Normal, {'loc': 3, 'scale': 0.01}, transform=var_tf)

    mb.add_latent('a0', nom='a0_nom', dev='a0_dev', chp='a0_chp', lot='a0_lot')
    mb.add_latent('e_aa', nom='e_aa_nom', dev=None, chp=None, lot=None)

    mb.add_intermediate('dvth', bti_vth_shift_empirical)

    mb.add_observed('dvth_meas', dists.Normal, {'loc': 'dvth', 'scale': 'meas_var'}, 3)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9861823450)
    basic_model = am.relmdl.spm

    def basic_sample(k, dims, priors, conds, prms):
        return trace(seed(basic_model, k)).get_trace(dims, conds, priors, prms)['e_dvth_meas']['value']

    pri = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
           'a0_dev': {'loc': 7, 'scale': 2},
           'a0_chp': {'loc': 5, 'scale': 2},
           'a0_lot': {'loc': 5, 'scale': 2},
           'e_aa_nom': {'loc': 5, 'scale': 2}}

    d = TestDef('t1', {'e': {'lot': 4, 'chp': 2}}, {'e': {'temp': 55 + 273.15, 'vdd': 0.88, 'time': 1000}})
    prms = am.relmdl.param_vals

    to_time = partial(basic_sample, k1, d.dims, pri, d.conds, prms)
    best_perf = min(timeit.Timer(to_time).repeat(repeat=100, number=100))
    print(f'Best unjitted: {best_perf}')

    # Now jit compile it
    j_sample = jax.jit(basic_sample, static_argnames='dims')
    # Run it once with the correct static args to compile before timing
    print(j_sample(k1, d.dims, pri, d.conds, prms))

    to_time = partial(j_sample, k1, d.dims, pri, d.conds, prms)
    best_perf = min(timeit.Timer(to_time).repeat(repeat=100, number=100))
    print(f'Best jitted: {best_perf}')




if __name__ == '__main__':
    main()
