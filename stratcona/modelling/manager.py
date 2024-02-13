# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

# I think it makes the most sense to have two model-level classes within Stratcona. The first is the model builder which
# needs to set everything up in order to spit out the needed PyMC model, but this requires a bunch of functionality and
# data which is no longer needed as soon as the model is built. To simplify the user experience, a model manager class
# here can be used to track the still-needed information and quickly call the various operations that can be performed
# on a model to simplify the user's experience in actually manipulating or computing quantities using the model.

from datetime import timedelta
import pytensor as pt
import pymc
from matplotlib import pyplot as plt

from gerabaldi.models.reports import TestSimReport
import gracefall

from stratcona.assistants.probability import shorthand_compile # noqa: ImportNotAtTopOfFile
from stratcona.engine.inference import *
from stratcona.engine.boed import *
from stratcona.engine.metrics import *
from stratcona.modelling.builder import ModelBuilder


class TestDesignManager:
    def __init__(self, model_builder: ModelBuilder):
        self.design_name = None
        self._test_model, self.latents_info, self.observed_info, self.predictor_info = model_builder.build_model()

        self._handlers = {}
        self._maps = {}
        self._handlers['exp'] = model_builder.experiment_handle
        self._handlers['pri'] = model_builder.priors_handle
        self._handlers['obs'] = model_builder.observation_handle
        self._maps['exp'] = model_builder.experiment_map
        self._maps['pri'] = model_builder.priors_map
        self._compiled_funcs = {}

    def determine_best_test(self, exp_sampler, obs_range):
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

        # TODO: The biggest inefficiency of the sampling BOED method used is that many obs samples will be completely
        #       unrealistic for the given test. Can we have the obs_sampler take the experiment as input?!?! This could
        #       make the value of a given sample much higher
        def obs_sampler():
            centre_vals = np.random.uniform(obs_range[0], obs_range[1], size=(2, 2))
            #all_vals = np.random.normal(centre_vals, (obs_range[1] - obs_range[0]) / 5)
            return centre_vals
        eigs = boed_runner(10, 50, 50, exp_sampler, self._handlers['exp'], self._compiled_funcs['ltnt_sampler'],
                           obs_sampler, self._compiled_funcs['ltnt_logp'], self._compiled_funcs['obs_logp'])

        # Optionally plot them all
        # TODO: Sort the designs for plotting somehow?!
        gracefall.static.eig_plot(eigs)
        plt.show()

        best_test = eigs.iloc[eigs['eig'].idxmax()]
        # Extract the best EIG test
        # TODO: Find a way to map experiments to some identifying names so that plotting and such are less cluttered
        print(f"Best experiment: {best_test['design']} with EIG of {best_test['eig']} nats")

    def infer_model(self, observations):
        self._handlers['obs'].set_observed(observations)
        idata = inference_model(self._test_model, num_samples=3000)
        posterior_prms = fit_latent_params_to_posterior_samples(self.latents_info, idata)
        self._handlers['pri'].set_params(posterior_prms)

    def set_experiment_conditions(self, conditions):
        self._handlers['exp'].set_experimental_params(conditions)

    def set_priors(self, priors):
        self._handlers['pri'].set_params(priors)

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
                gv = pymc.model_to_graphviz(self._test_model)
                gv.format = 'png'
                gv.render(filename='model_graph')
                plt.show()
                if not 'obs_sampler' in self._compiled_funcs.keys():
                    self._compiled_funcs['obs_sampler'] = \
                        shorthand_compile('obs_sampler', self._test_model, self.latents_info, self.observed_info, self.predictor_info)
                    pt.printing.pydotprint(self._compiled_funcs['obs_sampler'], 'func.png')
                # Use a Gerabaldi report for passing to visualization generators
                report = TestSimReport(name='prior_predictive')

                # Generate the samples
                sampled = [self._compiled_funcs['obs_sampler']() for _ in range(num_samples)]
                sampled = np.array(sampled)
                num_experiments = sampled.shape[2]
                sampled = np.split(sampled, num_experiments, axis=2)
                for exp in range(num_experiments):
                    # TODO: Change gerabaldi reports to avoid having to double array wrap the sampled values
                    exp_samples = sampled[exp].reshape((1, 1, -1))
                    as_dataframe = TestSimReport.format_measurements(exp_samples, f"exp{exp}", timedelta(), 0)
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
