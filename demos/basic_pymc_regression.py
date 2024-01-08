# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pymc
import arviz

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stratcona.modelling.builder import LatentVariable # noqa: ImportNotAtTopOfFile
from stratcona.engine.inference import inference_model, fit_latent_params_to_posterior_samples # noqa: ImportNotAtTopOfFile


def do_inference():
    if not os.path.exists('lin_regress.nc'):
        RANDOM_SEED = 8927
        rng = np.random.default_rng(RANDOM_SEED)

        size = 200
        true_intercept = 1
        true_slope = 2

        x = np.linspace(0, 1, size)
        # y = a + b*x
        true_regression_line = true_intercept + true_slope * x
        # add noise
        y = true_regression_line + rng.normal(scale=0.5, size=size)

        data = pd.DataFrame(dict(x=x, y=y))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
        ax.plot(x, y, "x", label="sampled data")
        ax.plot(x, true_regression_line, label="true regression line", lw=2.0)
        plt.legend(loc=0)

        with pymc.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
            # Define priors
            sigma = pymc.HalfCauchy("sigma", beta=10)
            intercept = pymc.Normal("intercept", 0, sigma=20)
            slope = pymc.Normal("slope", 0, sigma=20)

            # Define likelihood
            likelihood = pymc.Normal("y", mu=intercept + slope * x, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        idata = inference_model(model, num_samples=3000)
        idata.to_netcdf('lin_regress.nc')
    else:
        idata = arviz.from_netcdf('lin_regress.nc')

    latent_vars = {'sigma': LatentVariable('sigma', pymc.HalfCauchy, {'beta': 10}),
                   'intercept': LatentVariable('intercept', pymc.Normal, {'mu': 0, 'sigma': 20}),
                   'slope': LatentVariable('slope', pymc.Normal, {'mu': 0, 'sigma': 20})}
    new_prms = fit_latent_params_to_posterior_samples(latent_vars, idata, run_fit_analysis=True)
    #arviz.plot_trace(idata)
    #plt.show()


if __name__ == '__main__':
    do_inference()
