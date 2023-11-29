# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

# I think it makes the most sense to have two model-level classes within Stratcona. The first is the model builder which
# needs to set everything up in order to spit out the needed PyMC model, but this requires a bunch of functionality and
# data which is no longer needed as soon as the model is built. To simplify the user experience, a model manager class
# here can be used to track the still-needed information and quickly call the various operations that can be performed
# on a model to simplify the user's experience in actually manipulating or computing quantities using the model.
