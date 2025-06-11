
import numpyro as npyro
import numpyro.distributions as dists
from numpyro.handlers import seed, trace, condition

npyro.set_host_device_count(4)

import jax.numpy as jnp
import jax.random as rand
import jax

from dataclasses import dataclass
from functools import partial
import timeit
import time
import json

import seaborn
import matplotlib.pyplot as plt
import matplotlib.lines as pltlines
import pandas as pd

import numpyro.distributions as dist

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona
from stratcona.modelling.relmodel import TestDef, ExpDims
from stratcona.engine.inference import int_out_v
from stratcona.engine.minimization import minimize_jax


def dissertation_dist_transform_figure():
    mb = stratcona.SPMBuilder(mdl_name='VarTransform')

    var_tf = dist.transforms.ComposeTransform([dist.transforms.SoftplusTransform(), dist.transforms.AffineTransform(0, 0.1)])
    mb.add_params(obs_var=0.001)
    mb.add_hyperlatent('x1', dists.SoftLaplace, {'loc': 1.3, 'scale': 0.3})
    mb.add_observed('y1', dists.Normal, {'loc': 'x1', 'scale': 'obs_var'}, 1)
    mb.add_hyperlatent('x2', dists.SoftLaplace, {'loc': 1.3, 'scale': 0.3}, transform=dist.transforms.SoftplusTransform())
    mb.add_observed('y2', dists.Normal, {'loc': 'x2', 'scale': 'obs_var'}, 1)
    mb.add_hyperlatent('x3', dists.SoftLaplace, {'loc': 13, 'scale': 3.0}, transform=var_tf)
    mb.add_observed('y3', dists.Normal, {'loc': 'x3', 'scale': 'obs_var'}, 1)
    mb.add_hyperlatent('x4', dists.SoftLaplace, {'loc': 13, 'scale': 3.0}, transform=dist.transforms.SoftplusTransform())
    mb.add_observed('y4', dists.Normal, {'loc': 'x4', 'scale': 'obs_var'}, 1)

    print(dists.SoftLaplace(1.3, 0.3).variance)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=26523646)
    am.set_test_definition(TestDef('blank', {'e': {'lot': 1, 'chp': 1}}, {'e': {}}))

    vals = am.sim_test_meas_new((1_000_000,))
    m1, m2, m3 = jnp.mean(vals['e_y1']), jnp.mean(vals['e_y2']), jnp.mean(vals['e_y3'])
    print(m1)
    print(m2)

    # Now plot
    seaborn.set_context('notebook', font_scale=1.25)
    seaborn.set_theme(style='ticks', font='Times New Roman')
    fig, p = plt.subplots()
    p.hist(vals['e_y1'].flatten(), bins=200, density=True, histtype='stepfilled', color='mediumblue', alpha=1.0, label='No transform')
    p.hist(vals['e_y2'].flatten(), bins=200, density=True, histtype='stepfilled', color='darkviolet', alpha=0.8, label='Softplus only')
    p.hist(vals['e_y3'].flatten(), bins=200, density=True, histtype='stepfilled', color='deeppink', alpha=0.7, label='Softplus + affine')
    p.axvline(float(m1), color='mediumblue', linestyle='dotted')
    p.axvline(float(m2), color='darkviolet', linestyle='dotted')
    p.axvline(float(m3), color='deeppink', linestyle='dotted')
    p.annotate('Target Distribution:', (-0.35, 1.2), fontweight='bold')
    p.annotate('Soft Laplace\nlocation=1.3, scale=0.3, domain=(0, ∞)', (-0.35, 1.0))

    p.set_xlim([-0.5, 3])
    p.set_xlabel('Variable Value')
    p.set_ylabel('Probability Density')
    p.set_yticks([])

    hndl, lbls = p.get_legend_handles_labels()
    lgnd1 = pltlines.Line2D([0], [0], color='black', linestyle='dotted')
    hndl.insert(3, lgnd1)
    lbls.insert(3, 'Distribution means')

    # Add the custom legend
    leg = p.legend(hndl, lbls, loc='upper right')
    for lbl in leg.legend_handles:
        lbl.set_alpha(1)

    plt.show()


def experimenting():
    k, k2 = rand.split(rand.key(564637))
    vals = dist.LogNormal(3, 0.4).sample(k, (100_000,))

    fig, p = plt.subplots()
    p.hist(vals, histtype='stepfilled')
    plt.show()


if __name__ == '__main__':
    experimenting()
