# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0


def pymc_to_scipy(dist_type):
    match dist_type:
        case 'normal':
            return 'norm'
        case _:
            return dist_type
