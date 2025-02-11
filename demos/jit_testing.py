
import numpyro as npyro
import numpyro.distributions as dists
from numpyro.handlers import seed, trace, condition

import jax.numpy as jnp
import jax.random as rand
import jax

from dataclasses import dataclass
from functools import partial
import timeit
import time

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona
from stratcona.modelling.relmodel import TestDef, ExpDims


def jit_behaviour():

    @partial(jax.jit, static_argnames=['op'])
    def sw_f(x, y, op):
        if op == 'add':
            return x + y
        elif op == 'sub':
            return x - y
        else:
            return x * y

    @partial(jax.jit, static_argnames=['f'])
    def reduce(vals, f):
        res = vals[0]
        for i in range(1, len(vals)):
            res = f(res, vals[i])
        return res

    myarr = jnp.array([2,3,4,5,6])

    print(f"Call: {reduce(myarr, partial(sw_f, op='add'))}")
    print(f"Call: {reduce(myarr, partial(sw_f, op='sub'))}")
    print(f"Call: {reduce(myarr, partial(sw_f, op='mul'))}")


def main():
    key = rand.key(363)
    k1, k2, k3 = rand.split(key, 3)

    ####################################################
    # Define an NBTI empirical model within the SPM framework for inference
    ####################################################
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb = stratcona.SPMBuilder(mdl_name='bti-empirical')
    mb.add_params(k=8.617e-5, zero=0.0, meas_var=1, n_nom=2, alpha=3.5, n=2)

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

    pri = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
           'a0_dev': {'loc': 7, 'scale': 2},
           'a0_chp': {'loc': 5, 'scale': 2},
           'a0_lot': {'loc': 5, 'scale': 2},
           'e_aa_nom': {'loc': 5, 'scale': 2}}
    d = TestDef('t1', {'e': {'lot': 4, 'chp': 2}}, {'e': {'temp': 55 + 273.15, 'vdd': 0.88, 'time': 1000}})

    am.set_test_definition(d)
    am.relmdl.hyl_beliefs = pri
    s_f = am.relmdl.sample_new
    y_s = s_f(k2, d.dims, d.conds, (), am.relmdl.observes)
    y = {'e': {'dvth_meas': y_s['e_dvth_meas']}}
    am.do_inference_mhgibbs(y, beta=0.25)
    print(am.relmdl.hyl_beliefs)

    #x_s = s_f(k1, d.dims, d.conds, (100,), am.relmdl.hyls)
    #y_s = s_f(k2, d.dims, d.conds, (1,), am.relmdl.observes)
    #get_lp_y_g_x = stratcona.engine.inference.int_out_v
    #start = time.time()
    #lp_y_g_x, perf_stats = get_lp_y_g_x(k3, am.relmdl, (100, 100, 1), d.dims, d.conds, x_s, y_s)
    #print(f'Unjitted: {time.time() - start}s')
    #to_time = partial(get_lp_y_g_x, k3, am.relmdl, (100, 100, 1), d.dims, d.conds, x_s, y_s)
    #best_perf = min(timeit.Timer(to_time).repeat(repeat=10, number=10))
    #print(f'Jitted: {best_perf}s')


if __name__ == '__main__':
    jit_behaviour()
