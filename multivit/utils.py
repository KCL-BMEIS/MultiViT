from typing import Callable

import math

import numpy as np
import torch
from torch import nn


def log_sequences_of_things(logger, value, preamble=None):
    if logger is None:
        log = lambda x: print(x)
    else:
        log = lambda x: logger.info(x)

    if preamble is None:
        preamble = []
    if not isinstance(preamble, list):
        preamble = [preamble]

    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            log_sequences_of_things(logger, v, preamble + [str(i)])
    elif isinstance(value, dict):
        for k, v in value.items():
            log_sequences_of_things(logger, v, preamble + [k])
    elif isinstance(value, (torch.Tensor, np.ndarray)):
        log(f"{', '.join(preamble)}: {value.shape}")
    else:
        log(f"{', '.join(preamble)}: {value}")


def config_equals(config1, config2, stop_on_missing=True, context=None, logger=None):
    if context is None:
        context = []
    logger.info(f"config_equals ({context}): {type(config1) if config1 else None} vs {type(config2) if config2 else None}")
    if type(config1) != type(config2):
        logger.info(f"Type mismatch: {type(config1)} vs {type(config2)}")
        return False
    if isinstance(config1, dict):
        logger.info(f"Only in config1: {config1.keys() - config2.keys()}")
        logger.info(f"Only in config2: {config2.keys() - config1.keys()}")
        var_values1 = config1
        var_values2 = config2
    else:
        vars1 = {a for a in vars(config1) if not a.startswith('__')}
        vars2 = {a for a in vars(config2) if not a.startswith('__')}
        logger.info(f"Only in config1: {vars1.difference(vars2)}")
        logger.info(f"Only in config2: {vars2.difference(vars1)}")
        shared_vars = vars1.intersection(vars2)
        var_values1 = {k: getattr(config1, k) for k in shared_vars}
        var_values2 = {k: getattr(config2, k) for k in shared_vars}
        if vars1 != vars2 and stop_on_missing:
            return False
    for k, v in var_values1.items():
        logger.info(f"Checking {k}")
        if type(v) != type(var_values2[k]):
            logger.info(f"Type mismatch: {type(v)} vs {type(var_values2[k])}")
            return False
        if v != var_values2[k]:
            logger.info(f"Key {k} has different value: {v} vs {var_values2[k]}")
            return False
        if isinstance(v, dict):
            if not config_equals(v, var_values2[k], stop_on_missing, context + [k], logger):
                logger.info(f"Key {k} has different value: {v} vs {var_values2[k]}")
                return False


def decimal_power_str_to_value(value):
    return math.pow(10, int(value) / 10)