# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

# I think it makes the most sense to have two model-level classes within Stratcona. The first is the model builder which
# needs to set everything up in order to spit out the needed PyMC model, but this requires a bunch of functionality and
# data which is no longer needed as soon as the model is built. To simplify the user experience, a model manager class
# here can be used to track the still-needed information and quickly call the various operations that can be performed
# on a model to simplify the user's experience in actually manipulating or computing quantities using the model.

from datetime import timedelta
import pytensor as pt

from gerabaldi.models.reports import TestSimReport
import gracefall

from stratcona.assistants.probability import shorthand_compile # noqa: ImportNotAtTopOfFile
from stratcona.engine.inference import *
from stratcona.engine.metrics import *
from stratcona.modelling.builder import ModelBuilder
from stratcona.modelling.tensor_dict_translator import *


class TestDesignManager:
    def __init__(self, model_builder: ModelBuilder):
        self.design_name = None
        self._test_model, self.latents_info, self.observed_info, self.predictor_info = model_builder.build_model()

        self._handlers = {}
        self._maps = {}
        self._handlers['exp'] = model_builder.experiment_handle
        self._handlers['pri'] = model_builder.priors_handle
        self._handlers['obs'] = model_builder.observed_handles
        self._maps['exp'] = model_builder.experiment_map
        self._maps['pri'] = model_builder.priors_map
        self._compiled_funcs = {}

    def determine_best_test(self):
        # First compile any of the required graph computations that have not already been done
        if not 'ltnt_sampler' in self._compiled_funcs.keys():
            self._compiled_funcs['ltnt_sampler'] =\
                shorthand_compile('ltnt_sampler', self._test_model, self.latents_info, self.observed_info, self.predictor_info)
        if not 'ltnt_logp' in self._compiled_funcs.keys():
            self._compiled_funcs['ltnt_logp'] =\
                shorthand_compile('ltnt_logp', self._test_model, self.latents_info, self.observed_info, self.predictor_info)
        if not 'obs_logp' in self._compiled_funcs.keys():
            self._compiled_funcs['obs_logp'] =\
                shorthand_compile('obs_logp', self._test_model, self.latents_info, self.observed_info, self.predictor_info)

        best_test = None
        self._handlers['exp'].set_value(best_test)

    def infer_model(self, observations):
        for var in self.observed_info:
            self._handlers['obs'][var.name].set_value(observations[var.name])
        idata = inference_model(self._test_model, num_samples=3000)
        self._handlers['pri'].set_value(fit_latent_params_to_posterior_samples(self.latents_info, idata))

    def set_experiment_conditions(self, conditions):
        as_tensor = translate_experiment(conditions, self._maps['exp'])
        self._handlers['exp'].set_value(as_tensor)

    def set_priors(self, priors):
        as_tensor = translate_priors(priors, self._maps['pri'])
        self._handlers['pri'].set_value(as_tensor)

    def set_observations(self, var_name, observed):
        self._handlers['obs'][var_name].set_value(observed)

    def estimate_reliability(self):
        if not 'life_sampler' in self._compiled_funcs.keys():
            self._compiled_funcs['life_sampler'] =\
                shorthand_compile('life_sampler', self._test_model, self.latents_info, self.observed_info, self.predictor_info)
        bound = worst_case_quantile_credible_region(self._compiled_funcs['life_sampler'], 90)
        return bound

    def examine(self, attribute: str, num_samples: int = 300):
        """
        Since the whole process of using Stratcona is complex, it's very helpful to be able to look at how the model
        is behaving. This function helps generate some plots of different quantities so users can see what their model
        is doing.

        Parameters
        ----------
        attribute: The aspect of the model to look at
        """
        match attribute:
            case 'prior_predictive':
                if not 'obs_sampler' in self._compiled_funcs.keys():
                    self._compiled_funcs['obs_sampler'] = \
                        shorthand_compile('obs_sampler', self._test_model, self.latents_info, self.observed_info, self.predictor_info)
                    pt.printing.pydotprint(self._compiled_funcs['obs_sampler'], 'func.png')
                # Turn into a Gerabaldi report for passing to visualization generators
                report = TestSimReport(name='prior_predictive')
                # Generate the samples
                sampled = [self._compiled_funcs['obs_sampler']() for _ in range(num_samples)]
                # TODO: Change gerabaldi reports to avoid having to double array wrap the sampled values
                sampled = np.array(sampled).reshape((1, 1, -1))
                as_dataframe = TestSimReport.format_measurements(sampled, 'observed', timedelta(), 0)
                report.add_measurements(as_dataframe)
                gracefall.static.gen_stripplot_generic(report.measurements)
            case 'latents':
                if not 'ltnt_sampler' in self._compiled_funcs.keys():
                    self._compiled_funcs['ltnt_sampler'] =\
                        shorthand_compile('ltnt_sampler', self._test_model, self.latents_info, self.observed_info, self.predictor_info)
                # Turn into a Gerabaldi report for passing to visualization generators
                report = TestSimReport(name='prior_predictive')
                sampled = np.array([self._compiled_funcs['ltnt_sampler']() for _ in range(num_samples)])
                for i, ltnt in enumerate(self.latents_info):
                    # The stack is required as PyMC spits out different dimensionalities depending on distribution
                    ltnt_sampled = np.stack(sampled[:, i])
                    ltnt_sampled = ltnt_sampled.flatten().reshape((1, 1, -1))
                    as_dataframe = TestSimReport.format_measurements(ltnt_sampled, ltnt.name, timedelta(), 0)
                    report.add_measurements(as_dataframe)
                gracefall.static.gen_stripplot_generic(report.measurements)
            case _:
                raise NotImplementedError()
