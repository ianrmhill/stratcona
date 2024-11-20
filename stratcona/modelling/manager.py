# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

# I think it makes the most sense to have two model-level classes within Stratcona. The first is the model builder which
# needs to set everything up in order to spit out the needed PyMC model, but this requires a bunch of functionality and
# data which is no longer needed as soon as the model is built. To simplify the user experience, a model manager class
# here can be used to track the still-needed information and quickly call the various operations that can be performed
# on a model to simplify the user's experience in actually manipulating or computing quantities using the model.

from datetime import timedelta
from functools import partial
import pytensor as pt
import pymc
from matplotlib import pyplot as plt

import jax.random as rand

from gerabaldi.models.reports import TestSimReport
import gracefall

from stratcona.assistants.probability import shorthand_compile # noqa: ImportNotAtTopOfFile
from stratcona.engine.inference import inference_model
from stratcona.engine.boed import *
from stratcona.engine.metrics import *
from stratcona.modelling.builder import SPMBuilder
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

    def sim_test_measurements(self, alt_priors=None, rtrn_tr=False):
        rng = self._derive_key()
        if rtrn_tr:
            measd = self.relmdl.sample(rng, self.test)
        else:
            measd = self.relmdl.sample_measurements(rng, self.test.config, self.test.conditions, priors=alt_priors)
        return measd

    def do_inference(self, observations, test: ReliabilityTest = None, auto_update_prior=True):
        rng = self._derive_key()
        test_info = test if test is not None else self.test
        inf_mdl = partial(self.relmdl.test_spm, test_info.config, test_info.conditions, self.relmdl.hyl_beliefs, self.relmdl.param_vals)
        new_prior = inference_model(inf_mdl, self.relmdl.hyl_info, observations, rng)
        if auto_update_prior:
            self.relmdl.hyl_beliefs = new_prior

    def evaluate_reliability(self, predictor, num_samples=300_000):
        rng = self._derive_key()
        sampler = partial(self.relmdl.sample_predictor_from_beliefs, rng, predictor)

        if self.relreq.type == 'lbci':
            lifespan = worst_case_quantile_credible_region(sampler, self.relreq.quantile, num_samples)
        else:
            raise Exception('Invalid metric type')

        if lifespan >= self.relreq.target_lifespan:
            print(f'Target lifespan of {self.relreq.target_lifespan} met! Predicted: {lifespan}.')
        else:
            print(f'Target lifespan of {self.relreq.target_lifespan} not met! Predicted: {lifespan}.')
