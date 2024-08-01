#!/usr/bin/env python3

import os
import sys
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import traceback

import pandas as pd
from multivit import parsers, training, utils


head_configs = {
    'whole_pasi_only_linear': {
        'name': 'whole_pasi_linear',
        'label': 'n_pasi_0',
        'head': {'head': 'linear', 'kwargs': {}},
        'loss': {'loss': 'mse', 'kwargs': {}},
    },
    'whole_pasi_only_mlp_1': {
        'name': 'whole_pasi_mlp_1',
        'label': 'n_pasi_0',
        'head': {'head': 'mlp', 'kwargs': {}},
        'loss': {'loss': 'mse', 'kwargs': {}},
    },
    'severity_only_linear': {
        'name': 'severity_linear',
        'label': 'severity',
        'head': {'head': 'linear', 'nclasses': 5},
        'loss': {'loss': 'xent'},
    },
    'severity_only_mlp_1': {
        'name': 'severity_mlp_1',
        'label': 'severity',
        'head': {'head': 'mlp', 'classes': 5},
        'loss': {'loss': 'xent'},
    },
    'severity_plus_whole_pasi_0.5_linear': [
        {
            'name': 'whole_pasi_linear',
            'label': 'n_pasi_0',
            'head': {'head': 'linear', 'kwargs': {}},
            'loss': {'loss': 'mse', 'label': "n_pasi_0", 'weight': 0.5},
        },
        {
            'name': 'severity_linear',
            'label': 'severity',
            'head': {'head': 'linear', 'classes': 5},
            'loss': {'loss': 'xent', 'label': "physician_category", 'weight': 0.5},
        }
    ],
    'severity_plus_whole_pasi_0.5_mlp_1': [
        {
            'name': 'whole_pasi_mlp_1',
            'label': 'n_pasi_0',
            'head': {'head': 'mlp', 'kwargs': {'hidden_layers': 1, 'dropout': 0, 'input_channels': 1024, 'hidden_channels': 512, 'output_channels': 1}},
            'loss': {'loss': 'mse', 'label': "n_pasi_0", 'weight': 0.5},
        },
        {
            'name': 'severity_mlp_1',
            'label': 'severity',
            'head': {'head': 'mlp', 'kwargs': {'hidden_layers': 1, 'dropout': 0.1, 'hidden_channels': 512, 'input_channels': 1024, 'output_channels': 5}, 'classes': 5},
            'loss': {'loss': 'xent', 'label': "physician_category", 'weight': 0.5},
        }
    ],
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--base_dir', default=os.getcwd())
    parser.add_argument('-v', '--version', required=True)
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-m', '--model', required=False, default=None)
    parser.add_argument('-i', '--images', required=True)
    parser.add_argument('-n', '--network_mode', nargs='+', required=True)
    parser.add_argument('-u', '--use_subset', default='all')
    parser.add_argument('-p', '--augment', nargs='+', required=False, default=None)
    adaptor_group = parser.add_mutually_exclusive_group()
    adaptor_group.add_argument('-a', '--adaptor', action='append', nargs='+', required=False, default=None)
    adaptor_group.add_argument('-A', '--adaptor_config', required=False, default=None)
    parser.add_argument('-l', '--learning_rate', required=False, default=-30, type=int)
    parser.add_argument('-c', '--cache', action='store_true')
    parser.add_argument('--weight_decay', required=False, default=0, type=int)
    parser.add_argument('--fine_tune', nargs='+', required=False, default='probe')
    parser.add_argument('--hidden_dropout', required=False, default=0, type=int)
    parser.add_argument('--NOLOG', action='store_true')

    parsed_args = parser.parse_args()

    dataset_path = os.path.join(parsed_args.base_dir, parsed_args.dataset)
    dataset = pd.read_hdf(parsed_args.dataset)
    model_name = 'google-vit-large-patch32-384' if parsed_args.model is None else parsed_args.model
    model_path = os.path.join(parsed_args.base_dir, 'models', model_name)

    if parsed_args.adaptor_config is None:
        parsed_args.adaptor = ['linear'] if len(parsed_args.adaptor) == 0 else parsed_args.adaptor
    parsed_args.fine_tune = parsed_args.fine_tune if isinstance(parsed_args.fine_tune, list) else [parsed_args.fine_tune]

    trainer = training.training_functions.get(parsed_args.version, None)

    try:
        if trainer is not None:
            kwargs = {
                'base_dir': parsed_args.base_dir,
                'dataset': dataset,
                'model_path': model_path,
                'cache_data': parsed_args.cache,
            }
            kwargs['images'] = parsed_args.images
            kwargs['network_mode'] = parsed_args.network_mode
            kwargs['use_subset'] = "all" if parsed_args.use_subset is None else parsed_args.use_subset
            kwargs['augment'] = parsed_args.augment
            kwargs['fine_tune'] = parsed_args.fine_tune
            if parsed_args.adaptor_config is not None:
                kwargs['adaptor'] = head_configs[parsed_args.adaptor_config]
            else:
                kwargs['adaptor'] = [parsers.parse_adaptor_arguments(p) for p in parsed_args.adaptor]
            kwargs['learning_rate'] = utils.decimal_power_str_to_value(parsed_args.learning_rate)
            kwargs['weight_decay'] = utils.decimal_power_str_to_value(parsed_args.weight_decay)
            kwargs['hidden_dropout'] = utils.decimal_power_str_to_value(parsed_args.hidden_dropout)
            kwargs['log_to_file'] = not parsed_args.NOLOG

            file_name = (
                "model_"
                f"DS{parsed_args.use_subset}_"
                f"C{parsed_args.cache}_"
                f"N{'-'.join(n for n in parsed_args.network_mode) if isinstance(parsed_args.network_mode, (list, tuple)) else parsed_args.network_mode}_"
                f"FT{'-'.join(f for f in parsed_args.fine_tune)}_"
                f"A{'-'.join(b for a in parsed_args.adaptor for b in a) if parsed_args.adaptor is not None else parsed_args.adaptor_config}_"
                f"L{parsed_args.learning_rate}_"
                f"WD{parsed_args.weight_decay}_"
                f"HD{parsed_args.hidden_dropout}_"
                f"AUG{'No' if parsed_args.augment is None else '-'.join(a for a in parsed_args.augment)}"
            )
            kwargs['file_name'] = file_name

            training.train_v0_1(**kwargs)
        else:
            raise ValueError(f"'version' must be one of {tuple(k for k in training.training_functions.keys())} but is {parsed_args.version}")
    except Exception as e:
        print(e)
        exc_info = sys.exc_info()
        print(traceback.print_exception(*exc_info))


if __name__ == '__main__':
    main()
