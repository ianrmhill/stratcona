
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


def hmc_viz():
    f1 = lambda x, y: jnp.sin((6-(0.2*x))*(x+(0.2*y)) + 1)
    #f2 = lambda x, y: (1.3 - (0.05*x))**(jnp.abs(y-4))
    f2 = lambda x, y: 1.3**(jnp.abs(y-5))

    x = jnp.linspace(0, 10, 100)
    xt = jnp.repeat(jnp.expand_dims(x, 1), 100, axis=1)
    y = jnp.linspace(0, 10, 100)
    yt = jnp.repeat(jnp.expand_dims(y, 0), 100, axis=0)
    lp = f1(xt, yt) + f2(xt, yt)

    # Now plot
    seaborn.set_context('notebook', font_scale=1.25)
    seaborn.set_theme(style='ticks', font='Times New Roman')
    fig, p = plt.subplots()

    lpc = p.contourf(x, y, lp,
                      levels=30, cmap='inferno')
    # vlin = tc.where(tc.lt(tc.tensor(0), v2 - v1) & tc.gt(tc.tensor(1), v2 - v1), v2 - v1, -1)
    #p.plot(v1, v2, color='black', linestyle='', marker='.', label='Discrete test designs')
    p.set_xlabel('x1')
    p.set_ylabel('x2')
    # Set up legend
    cbar = fig.colorbar(lpc)
    cbar.ax.set_ylabel('-log(p(x|y))')
    p.legend(loc='lower right', fontsize=14)

    p.set_ylim([0, 5])
    #p.set_xlabel('Variable Value')
    #p.set_ylabel('Probability Density')
    #p.set_yticks([])

    #hndl, lbls = p.get_legend_handles_labels()
    #lgnd1 = pltlines.Line2D([0], [0], color='black', linestyle='dotted')
    #hndl.insert(3, lgnd1)
    #lbls.insert(3, 'Distribution means')

    # Add the custom legend
    #leg = p.legend(hndl, lbls, loc='upper right')
    #for lbl in leg.legend_handles:
    #    lbl.set_alpha(1)

    plt.show()

if __name__ == '__main__':
    hmc_viz()
