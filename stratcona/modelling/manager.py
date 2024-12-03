# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from functools import partial
from matplotlib import pyplot as plt

import jax.random as rand

from gerabaldi.models.reports import TestSimReport
import gracefall

from stratcona.engine.inference import inference_model
from stratcona.engine.bed import bed_run
from stratcona.engine.metrics import *
from stratcona.modelling.relmodel import ReliabilityModel, ReliabilityTest, ReliabilityRequirement


class AnalysisManager:
    def __init__(self, model: ReliabilityModel, rng_seed: int, rel_req: ReliabilityRequirement = None):
        self.relmdl = model
        self.relreq = rel_req
        self.test = None
        self.rng_key = rand.key(rng_seed)

    def _derive_key(self):
        self.rng_key, for_use = rand.split(self.rng_key)
        return for_use

    def set_test_definition(self, test_def: ReliabilityTest):
        self.test = test_def

    def update_priors(self, new_priors):
        self.relmdl.hyl_beliefs = new_priors

    def examine_model(self, model_components: list):
        rng = self._derive_key()

        meas_sites = []
        for exp in self.test.config:
            if 'meas' in model_components:
                meas_sites.extend([f'{meas}_{exp}' for meas in self.relmdl.test_measurements])
            if 'ltnts' in model_components:
                for meas in self.relmdl.test_measurements:
                    meas_sites.extend([f'{ltnt}_{meas}_{exp}' for ltnt in self.relmdl.ltnts])
        if 'hyls' in model_components:
            meas_sites.extend(self.relmdl.hyls)

        measd = self.relmdl.sample(rng, self.test, num_samples=50_000, keep_sites=meas_sites)

        for site in measd:
            report = TestSimReport(name='Sampled Values')
            exp_samples = measd[site].reshape((1, 1, -1))
            print(f'Mean of {site}: {jnp.mean(exp_samples)}')
            as_dataframe = TestSimReport.format_measurements(exp_samples, site, timedelta(), 0)
            report.add_measurements(as_dataframe)
            gracefall.static.gen_violinplot(report.measurements.loc[report.measurements['param'] == site])

    def sim_test_measurements(self, alt_priors=None, rtrn_tr=False, num=()):
        rng = self._derive_key()
        return self.relmdl.sample(rng, self.test, num_samples=num, alt_priors=alt_priors, full_trace=rtrn_tr)

    def do_inference(self, observations, test: ReliabilityTest = None, auto_update_prior=True):
        rng = self._derive_key()
        test_info = test if test is not None else self.test
        inf_mdl = partial(self.relmdl.test_spm, test_info.config, test_info.conditions, self.relmdl.hyl_beliefs, self.relmdl.param_vals)
        new_prior = inference_model(inf_mdl, self.relmdl.hyl_info, observations, rng)
        if auto_update_prior:
            self.relmdl.hyl_beliefs = new_prior

    def evaluate_reliability(self, predictor, num_samples=300_000, plot_results=False):
        rng = self._derive_key()
        samples = self.relmdl.sample(rng, self.test, (num_samples,), keep_sites=predictor)

        lifespan = self.relreq.type(samples, self.relreq.quantile)

        if lifespan >= self.relreq.target_lifespan:
            print(f'Target lifespan of {self.relreq.target_lifespan} met! Predicted: {lifespan}.')
        else:
            print(f'Target lifespan of {self.relreq.target_lifespan} not met! Predicted: {lifespan}.')

        if plot_results:
            fig, p = plt.subplots(1, 1)
            p.hist(samples, 300, density=True, color='grey', histtype='stepfilled')
            p.axvline(lifespan, 0, 1, color='orange', linestyle='solid',
                      label=f'Q{self.relreq.quantile}%-LBCI: {round(float(lifespan), 2)}')
            p.axvline(self.relreq.target_lifespan, 0, 1, color='orange', linestyle='dashed',
                      label=f'Target lifespan: {round(float(self.relreq.target_lifespan), 2)}')

            p.legend()
            p.set_xlabel('Failure Time (years)', fontsize='medium')
            p.set_ylabel('Probability Density')

    def find_best_experiment(self, l, n, m, exp_sampler):
        rng = self._derive_key()
        return bed_run(rng, l, n, m, exp_sampler, self.relmdl)
