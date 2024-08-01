import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import shutil
import copy
from datetime import datetime
import time
import logging
import json
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import datasets
from datasets import Dataset, load_metric
from peft import LoraConfig, get_peft_model
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback, ViTImageProcessor, ViTFeatureExtractor, ViTModel
from transformers import ViTForImageClassification, ViTConfig
from transformers.image_utils import PILImageResampling
from transformers.modeling_utils import unwrap_model
from transformers.utils import logging as tlogging

from multivit.config import MultiViTComponentConfig, MultiViTConfig
import multivit.dataset as mvds
import multivit.dataset_helpers as dh
from multivit.modules import (
    get_per_layer_output_policy,
    parse_network_mode,
    MultiLayerPerceptron,
    MultiViTComponent,
    MultiViT,
)
from multivit import parsers
import multivit.preprocessing as preproc
from multivit.utils import config_equals, log_sequences_of_things
import logging
from torch import nn



def decimal_negative_power_str_to_value(value):
    return math.pow(10, -int(value) / 10)



def prepare_model_and_processor(
        model_path,
        network_mode,
        verbose=False,
        fine_tune="none",
        adaptor=None,
        hidden_state_policy=None,
        attention_policy=None,
        hidden_dropout=None,
        **overrides
):
    logger = logging.getLogger(f"{__name__}.{prepare_model_and_processor.__name__}")

    attention = overrides.get("attention", False)

    if verbose:
        logger.info("fetching weights for processor")

    if verbose:
        logger.info("fetching weights for model")

    config: ViTConfig = ViTConfig.from_pretrained(model_path, output_attentions=True, output_hidden_states=True, local_files_only=True)
    hidden_state_by_layer = get_per_layer_output_policy(hidden_state_policy, config.num_hidden_layers)
    attention_by_layer = get_per_layer_output_policy(attention_policy, config.num_hidden_layers)
    logger.info(f"output_hidden_states={config.output_hidden_states}")
    logger.info(f"image_size={config.image_size}, patch_size={config.patch_size} patches={config.image_size / config.patch_size}")
    if hidden_dropout is not None:
        hidden_dropout_ = hidden_dropout
        logger.info(f"hidden_dropout_prob: set to {hidden_dropout} from default of {config.hidden_dropout_prob}")
    else:
        hidden_dropout_ = config.hidden_dropout_prob
        logger.info(f"hidden_dropout_prob: default of {config.hidden_dropout_prob}")

    logger.info(f"fine_tune: {fine_tune}")

    adaptor = [parsers.set_adaptor_sizes(config.hidden_size, a['head'].get('classes', 1), a) for a in adaptor]

    cmp_config_0 = MultiViTComponentConfig(
        net_config=copy.deepcopy(config),
        stage=0,
        weights_name=model_path,
        fine_tuning=fine_tune,
        adaptors=adaptor,
        output_hidden_states=True,
        output_attentions=True,
        output_hidden_states_by_layer=hidden_state_by_layer,
        output_attentions_by_layer=attention_by_layer,
    )
    network_mode_, image_count, resolution_multiplier = parse_network_mode(network_mode)
    if network_mode_ != "simple":
        cmp_config_1 = MultiViTComponentConfig(
            net_config=copy.deepcopy(config),
            weights_name=model_path,
            stage=1,
            fine_tuning=fine_tune,
            adaptors=adaptor,
            output_hidden_states=True,
            output_attentions=True,
            output_hidden_states_by_layer=hidden_state_by_layer,
            output_attentions_by_layer=attention_by_layer,
        )
        if network_mode_ == "multi":
            cmp_config_1.use_embeddings = True
        elif network_mode_ == "full":
            cmp_config_1.use_embeddings = resolution_multiplier > 1
            cmp_config_1.concat_embeddings = resolution_multiplier > 1

    else:
        cmp_config_1 = None

    mv_config = MultiViTConfig(
        network_mode=network_mode_,
        image_count=image_count,
        resolution_multiplier=resolution_multiplier,
        net_config_0=cmp_config_0,
        net_config_1=cmp_config_1,
    )

    if mv_config.resolution_multiplier != 1:
        size_hr = {
            "height": cmp_config_0.net_config.image_size * mv_config.resolution_multiplier,
            "width": cmp_config_0.net_config.image_size * mv_config.resolution_multiplier
        }
    else:
        size_hr = None

    logger.info(f"network_mode: {mv_config.network_mode}")

    model = MultiViT(config=mv_config)

    # TODO: it will be nice to be able to call this for the full model for inference
    # model = MultiViTComponent.from_pretrained(weights_name, config=mv_config)

    if fine_tune is not None:
        if fine_tune[0] == 'probe':
            for np, p in model.named_parameters():
                if 'vit.encoder.layer' in np:
                    p.requires_grad = False
        elif fine_tune[0] == 'last':
            # pass 1: find the number of layers in this architecture and find the first unfrozen index
            max_layer_index = None
            for np, p in model.named_parameters():
                if 'vit.encoder.layer' in np:
                    layer_index = int(np.split('.')[4])
                    if max_layer_index is None or layer_index > max_layer_index:
                        max_layer_index = layer_index
            first_unfrozen_index = max_layer_index - int(fine_tune[1]) + 1
            # pass 2: freeze all layers with a lower index than the first unfrozen index
            for np, p in model.named_parameters():
                if 'vit.encoder.layer' in np and int(np.split('.')[4]) < first_unfrozen_index:
                    p.requires_grad = False
        elif fine_tune[0] == 'peft':
            peft_config = LoraConfig(
                r=int(fine_tune[1]),
                lora_alpha=int(fine_tune[1]),
                target_modules=["query", "value"],
                lora_dropout=hidden_dropout_,
                bias="none",
                modules_to_save=[],
            )
            model = get_peft_model(model, peft_config)
        else:
            raise ValueError(f"Invalid mode '{fine_tune}'")

    for np, p in model.named_parameters():
        logger.info(f"{np}: {p.shape}: grad: {p.requires_grad}")

    if verbose:
        logger.info("---model---")
        logger.info(model)
        logger.info("---processor---")
    return model, mv_config



def transform_with_processor_v0_1(processor, data_path, augment=None):
    def _inner(batch, *args, **kwargs):
        input_images = [Image.open(os.path.join(data_path, img)) for img in batch['image_name']]
        if augment is not None:
            input_images = augment(input_images)
        images = processor(input_images, *args, **kwargs)
        rbatch = dict(batch)
        rbatch['pixel_values'] = images['pixel_values']
        if 'pixel_values_hr' in images:
            rbatch['pixel_values_hr'] = images['pixel_values_hr']
        return rbatch
    return _inner



def multi_split_dataset_v0_1(dataframe, data_path, cache_data, processor, validation_fold, images_per_sample, trn_processor, val_processor):
    logger = logging.getLogger("split_dataset_v0_1")
    split = ['trn' if f != validation_fold else 'val' for f in dataframe['fold']]
    dataframe['split'] = split
    dataframe['physician_category'] = dataframe['physician_category'] - 1
    print(dataframe['physician_category'].unique())

    trn_dataframe = dataframe[dataframe['split'] == 'trn']
    val_dataframe = dataframe[dataframe['split'] == 'val']
    trn_dataset = mvds.MultiViTDataset(
        mvds.CachedLoader(data_path) if cache_data is True else mvds.BasicLoader(data_path),
        trn_dataframe,
        images_per_sample,
        processor if trn_processor is None else trn_processor,
        None,
    )
    val_dataset = mvds.MultiViTDataset(
        mvds.CachedLoader(data_path) if cache_data is True else mvds.BasicLoader(data_path),
        val_dataframe,
        images_per_sample,
        processor if val_processor is None else val_processor,
        None,
    )

    logger.info(f"trn: {len(trn_dataset)}, val: {len(val_dataset)})")

    return trn_dataset, val_dataset



def split_dataset_v0_1(dataframe, data_path, cache_data, processor, validation_fold, trn_processor, val_processor):
    logger = logging.getLogger("split_dataset_v0_1")
    split = ['trn' if f != validation_fold else 'val' for f in dataframe['fold']]
    dataframe['split'] = split
    dataframe['physician_category'] = dataframe['physician_category'] - 1
    print(dataframe['physician_category'].unique())

    trn_dataframe = dataframe[dataframe['split'] == 'trn']
    val_dataframe = dataframe[dataframe['split'] == 'val']
    trn_dataset = mvds.MultiViTSimpleDataset(
        mvds.CachedLoader(data_path) if cache_data is True else mvds.BasicLoader(data_path),
        trn_dataframe,
        processor if trn_processor is None else trn_processor,
        None,
    )
    val_dataset = mvds.MultiViTSimpleDataset(
        mvds.CachedLoader(data_path) if cache_data is True else mvds.BasicLoader(data_path),
        val_dataframe,
        processor if val_processor is None else val_processor,
        None,
    )

    logger.info(f"trn: {len(trn_dataset)}, val: {len(val_dataset)})")

    return trn_dataset, val_dataset



def old_split_dataset_v0_1(dataframe, processor, validation_fold):
    logger = logging.getLogger("split_dataset_v0_1")
    split = ['trn' if f != validation_fold else 'val' for f in dataframe['fold']]
    dataframe['split'] = split
    dataframe['physician_category'] = dataframe['physician_category'] - 1

    dataset = Dataset.from_pandas(dataframe)
    trn_dataset = dataset.filter(lambda d: d['split'] == 'trn')
    val_dataset = dataset.filter(lambda d: d['split'] == 'val')
    logger.info(f"create training dataset (length {len(trn_dataset)})")
    logger.info(f"trn: {len(trn_dataset)}, val: {len(val_dataset)})")

    return trn_dataset, val_dataset



def collate_fn(batch):
    rbatch = dict()
    rbatch['pixel_values'] = torch.stack([x['pixel_values'] for x in batch])
    pixel_values_hr = [x.get('pixel_values_hr', None) for x in batch]
    if all(x is not None for x in pixel_values_hr):
        rbatch['pixel_values_hr'] = torch.stack(pixel_values_hr)
    mapping = [('total_0', 'pasi'), ('n_pasi_0', 'n_pasi_0'), ('physician_category', 'physician_category'),
               ('pid', 'pid'), ('iid', 'iid')]
    for src, dst in mapping:
        rbatch[dst] = torch.tensor([x[src] for x in batch])
        rbatch[dst] = rbatch[dst].unsqueeze(-1)
    return rbatch



def metrics(label_names, cache_location, experiment_id, attention=False):
    """
    2024-02-18 18:18:07,088|INFO|metrics|pasi: (116, 1)
    2024-02-18 18:18:07,089|INFO|metrics|pasi: 0, 2.4609375, 17.799999237060547
    2024-02-18 18:18:07,089|INFO|metrics|pasi: 1, 2.01171875, 17.799999237060547
    2024-02-18 18:18:07,089|INFO|metrics|pasi: 2, 1.9326171875, 17.799999237060547

    label_names is necessary because it defines the order for data.label_ids
    """
    # TODO: replace this with the MetricsCallback class below
    logger = logging.getLogger(f"{__name__}.{metrics.__name__}")
    metric = load_metric('mse', cache_dir=cache_location, experiment_id=experiment_id)
    logger.info(f"cache_location={cache_location}")
    logger.info(f"experiment_id={experiment_id}")
    def _inner(data):

        if len(data.predictions) == 4:
            # log the preliminary losses from the first network stage
            for k in label_names:
                for s in range(len(data.predictions[1])):
                    if k in data.predictions[1][s]['losses']:
                        vindex = label_names.index(k)
                        losses = data.predictions[1][s]['losses'][k]
                        logits = data.predictions[1][s]['logits'][k]
                        values = data.label_ids[vindex]
                        for i in range(len(losses)):
                            logger.info(f"{k}_0: {s}, {i}, {logits[i].item()}, {values[i].item()}, {losses[i].item()}")

        metric_result = dict()
        for k in label_names:
            if k in data.predictions[-1]['losses']:
                vindex = label_names.index(k)
                losses = data.predictions[-1]['losses'][k]
                logits = data.predictions[-1]['logits'][k]
                values = data.label_ids[vindex]
                for i in range(len(losses)):
                    logger.info(f"{k}: {i}, {logits[i].item()}, {values[i].item()}, {losses[i].item()}")
                metric_result[k] = metric.compute(predictions = logits.squeeze(), references=values.squeeze())
        return metric_result

    return _inner



class MetricsCallback(TrainerCallback):
    """
    This class allows us to control metric evaluation in a more fine-grained way than using
    a metric callback to `compute_metrics` on `Trainer`. It's job is to avoid the problem that
    metrics are calculated across all model outputs for the entire evaluation dataset. When expensive
    tensors like attention maps are stored for the entire evaluation dataset, they can take a great
    deal of memory:

    Es (number of eval dataset samples) = 512
    H (number of attention heads) = 16
    T (number of tokens) = 12 * 12 + 1 = 145
    Mem = 512 * 16 * 145 * 145 = 172,236,800 values * 8 bytes = 1,377,894,400 bytes
    """

    def __init__(self, metrics_names, metrics_ops):
        super().__init__()
        self.metrics = {m: None for m in metrics_names}
        self.metrics_ops = {m: o for m, o in zip(metrics_names, metrics_ops)}

    def on_evaluate(self, args, state, control, **kwargs):
        if control.should_evaluate is True:
            # evaluation start
            self.metrics = {k: None for k in self.metrics.keys()}
        else:
            # evaluation end
            logger = logging.get_logger()
            for k, v in self.metrics.items():
                op = self.metrics_ops[k]
                logger.info(f"{__class__}.{__name__}: {k}, {op(v)}")



class LoggingCallbacks(TrainerCallback):

    def on_step_begin(self, args, state, control, **kwargs):
        logger = logging.getLogger(f"{__class__.__name__}.{self.on_step_begin.__name__}")
        logger.info(f"step: {state.global_step}")
        for k, v in kwargs.items():
            logger.info(f"{k}")

    def on_step_end(self, args, state, control, **kwargs):
        logger = logging.getLogger(f"{__class__.__name__}.{self.on_step_end.__name__}")
        logger.info(f"step: {state.global_step}")
        for k, v in kwargs.items():
            logger.info(f"{k}")

    def on_prediction_step(self, args, state, control, **kwargs):
        logger = logging.getLogger(f"{__class__.__name__}.{self.on_prediction_step.__name__}")
        logger.info(f"step: {state.global_step}")
        for k, v in kwargs.items():
            logger.info(f"{k}")



class LoggingTrainer(Trainer):

    def __init__(self, validation_save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_marker = 0
        self.validation_save_path = validation_save_path
        self.validation_attentions_0 = list()
        self.validation_attentions_1 = list()
        self.mappings = list()
        self.mappings_hr = list()
        self.last_mode = None

    def detach_value_for_logging(self, value):
        if value.size() == 1:
            return value.item()
        else:
            return value.detach().tolist()


    def detach_to_list_or_scalar(self, value):
        if len(value.shape) == 0:
            return value.item()
        elif len(value.shape) == 1 and value.shape[0] == 1:
            return value.item()
        else:
            return value.detach().tolist()


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if model.training:
            mode = "training"
            if self.last_mode == "validation":
                self.save_validation_metrics()
        else:
            mode = "validation"
        self.last_mode = mode

        logger = logging.getLogger(f"{__class__.__name__}.{self.compute_loss.__name__}.{mode}")

        model_inputs = {k: v for k, v in inputs.items() if k in ('pixel_values', 'pixel_values_hr')}
        labels = {k: v for k, v in inputs.items() if k in ('n_pasi_0', 'pasi', 'physician_category')}
        model_inputs['labels'] = labels

        outputs = model(**model_inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later. (this is huggingface's TODO)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if 'loss_0' in outputs:
            loss_0 = outputs.get('loss_0', None)
            output_0 = outputs.get('output_0', dict())
            for k, cur_losses in output_0['losses'].items():
                for b_ind in range(cur_losses.shape[0]):
                    payload = {
                        "pid": self.detach_to_list_or_scalar(inputs["pid"][b_ind]),
                        "iid": self.detach_to_list_or_scalar(inputs["iid"][b_ind]),
                        "stage": 0,
                        "predict": self.detach_to_list_or_scalar(output_0['logits'][k][b_ind]),
                        "actual": self.detach_to_list_or_scalar(inputs[k][b_ind]),
                        "loss": self.detach_to_list_or_scalar(cur_losses[b_ind]),
                    }
                    logger.info(f"{k}: {payload}")
            loss = loss_0


            # attentions
            if mode == "validation":
                with torch.no_grad():
                    attentions = outputs['output_0']['attentions']
                    attentions = [a.detach() for a in attentions if a is not None]
                    class_attentions = [a[:, :, 0:1, :].cpu().numpy() for a in attentions]
                    self.validation_attentions_0.append(class_attentions)
        else:
            # add all the losses together
            losses_0 = outputs['losses_0']
            for s_ind in range(len(outputs['outputs_0'])):
                cur_outputs = outputs['outputs_0'][s_ind]
                for k, cur_losses in cur_outputs['losses'].items():
                    for b_ind in range(cur_losses.shape[0]):
                        payload = {
                            "pid": self.detach_to_list_or_scalar(inputs["pid"][s_ind, b_ind]),
                            "iid": self.detach_to_list_or_scalar(inputs["iid"][s_ind, b_ind]),
                            "stage": 0,
                            "set": s_ind,
                            "predict": self.detach_to_list_or_scalar(cur_outputs['logits'][k][b_ind]),
                            "actual": self.detach_to_list_or_scalar(inputs[k][s_ind, b_ind]),
                            "loss": self.detach_to_list_or_scalar(cur_losses[b_ind]),
                        }
                        logger.info(f"{k}: {payload}")
            loss_1 = outputs.get('loss_1', None)
            outputs_1 = outputs.get('output_1', dict())
            for k, cur_losses in outputs_1['losses'].items():
                for b_ind in range(cur_losses.shape[0]):
                    payload = {
                        "pid": self.detach_to_list_or_scalar(inputs["pid"][0, b_ind]),
                        "stage": 1,
                        "predict": self.detach_to_list_or_scalar(outputs_1['logits'][k][b_ind]),
                        "actual": self.detach_to_list_or_scalar(inputs[k][0, b_ind]),
                        "loss": self.detach_to_list_or_scalar(cur_losses[b_ind]),
                    }
                    logger.info(f"{k}: {payload}")

            loss = loss_1
            for loss_0 in losses_0:
                loss_0 = loss_0 / len(losses_0)
                loss = loss_0 if loss is None else loss + loss_0
            logger.info(f"loss: {loss}")

            # attentions
            if mode == "validation":
                with torch.no_grad():
                    all_class_attentions = list()
                    for s_ind in range(len(outputs['outputs_0'])):
                        attentions = outputs['outputs_0'][s_ind]['attentions']
                        attentions = [a.detach() for a in attentions if a is not None]
                        class_attentions = [a[:, :, 0:1, :].cpu().numpy() for a in attentions]
                        all_class_attentions.append(class_attentions)
                    self.validation_attentions_0.append(all_class_attentions)

                    attentions = outputs['output_1']['attentions']
                    attentions = [a.detach() for a in attentions if a is not None]
                    class_attentions = [a[:, :, 0:1, :].cpu().numpy() for a in attentions]
                    self.validation_attentions_1.append(class_attentions)

                    self.mappings.append(outputs['mappings'])
                    if 'mappings_hr' in outputs:
                        self.mappings_hr.append(outputs['mappings_hr'])

        return (loss, outputs) if return_outputs else loss


    def save_validation_metrics(self):
        series = dict()
        if len(self.validation_attentions_0) > 0:
            as_array = self.nested_stack(self.validation_attentions_0)
            series['validation_attentions_0'] = as_array
        if len(self.validation_attentions_1) > 0:
            as_array = self.nested_stack(self.validation_attentions_1)
            series['validation_attentions_1'] = as_array

        self.epoch_marker += 1
        os.makedirs(self.validation_save_path, exist_ok=True)
        np.savez(os.path.join(self.validation_save_path, f"validation_metrics_{self.epoch_marker:04}.npz"), **series)

        if len(self.mappings) > 0:
            with open(os.path.join(self.validation_save_path, f"mappings_{self.epoch_marker:04}.json"), 'w') as f:
                json.dump(self.mappings, f)
        if len(self.mappings_hr) > 0:
            with open(os.path.join(self.validation_save_path, f"mappings_hr_{self.epoch_marker:04}.json"), 'w') as f:
                json.dump(self.mappings_hr, f)

        self.validation_attentions_0 = list()
        self.validation_attentions_1 = list()
        self.mappings = list()
        self.mappings_hr = list()


    def nested_stack(self, list_of_tensors):
        if isinstance(list_of_tensors[0], list):
            pre_stacked_tensors = [self.nested_stack(t) for t in list_of_tensors]
            return np.stack(pre_stacked_tensors, axis=0)
        return np.stack(list_of_tensors, axis=0)


def train_v0_1(
        base_dir,
        dataset,
        model_path,
        cache_data,
        file_name,
        images=None,
        network_mode=None,
        use_subset=None,
        augment=None,
        fine_tune=None,
        adaptor=None,
        learning_rate=0.001,
        weight_decay=0,
        hidden_dropout=0,
        log_to_file=True,
):
    """

    The network can be configured to run in any of the following modes:
        - s: a single transformer
        - m_i_x: two transformer models executed in sequence. Each has its own losses and there is
          no passing of tokens from the first stage to the second stage
          - i is the number of images per sample for the first stage
          - x is the resolution multiplier to be applied for the second stage
        - f_i_x: two transformer models executed in sequence. Each has its own losses and the tokens
          selected by the first stage as passed to the second stage
          - i is the number of images per sample for the first stage
          - x is the resolution multipler to be applied for the second stage

    ## Adaptors:

    The network can be configured with one or more task adaptors. These take the final hidden layer
    (the class and patch tokens) learn a specific task. determined by the loss functions and labels involved. A task
    adaptor is defined as follows:

    ```
    adaptor_definition:
    {
        name: human readable id for adaptor,
        label: the name of the label from the inputs to test the adaptor against,
        head: head_definition | function
        classes: <int>, // only defined if the task is classification
        loss: [<loss_definition>] | <loss_definition>,
    }

    head_definition:
    {
        head: <head_type>,
        kwargs: {
            key: value, // 1+ key value pairs
        },
    }

    loss_definition:
    {
        label: <label_name>,
        weight: <float> (optional, 1.0 if not set),
        loss: <loss_type>,
        kwargs: {
            key: value, // 1+ key value pairs
        },
    }
    ```


    ## Fine tune:

    Networks can be configured for fine-tuning. By default, none of the network is frozen but the
    user can specify any of a number of fine-tuning policies:
      - the user can specify what layers of the network are tunable
        - freeze all: "probe"
        - first n: ["first", n]
        - last n: ["last", n]
      - the user can specify additional layers to be added to the network for tuning
        - add n: ["add", n] - not yet implemented
      - the user can specify peft-style fine-tuning strategies
        - peft: "lora"
          - rank: r, the reduced size for the A & B lora matrix dimensions

    Args:
        base_dir: the effective directory from which the model is being run
        dataset: the path and filename serialised pandas dataframe from which labels are loaded
        images: the path to the image files
        network_mode: the mode in which the network should be configured.
        use_subset: the subset of the dataset that should be used. If this isn't set, the full dataset
                    will be used
        augment: the augmentation that should be applied to the data for training
        fine_tune: the fine-tuning mode for the networks. Presently, the same fine tuning policy is
                   applied to both stages if the network_mode is m_*_* or f_*_*
        adaptor: a list of one or more task adaptors
    """
    attention = True

    logging_format_string = '%(asctime)s|%(levelname)s|%(name)s|%(message)s'

    stdout_handler = logging.StreamHandler(sys.stdout)

    use_subset_fn = dataset_filter_functions.get(use_subset, None)

    full_log_file_path = os.path.join(base_dir, 'runs', file_name)
    training_log_name = 'training.log'
    if log_to_file is True:
        os.makedirs(full_log_file_path)
        file_handler = logging.FileHandler(os.path.join(full_log_file_path, training_log_name))
        logging_handlers = [stdout_handler, file_handler]
    else:
        logging_handlers = [stdout_handler]

    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format_string,
        handlers=logging_handlers
    )

    tlogging.disable_progress_bar()

    logger = logging.getLogger("train_v0_1")

    logger.info("schema: {'name': 'multivit', 'version': 0.5}")
    logger.info(f"python: {shutil.which('python')}")

    logger.info(f"base_dir: {base_dir}")
    logger.info(f"model_path: {model_path}")
    logger.info(f"cache_data: {cache_data}")
    logger.info(f"file_name: {file_name}")
    logger.info(f"images: {images}")
    logger.info(f"network_mode: {network_mode}")
    logger.info(f"use_subset: {use_subset}")
    logger.info(f"augment: {augment}")
    logger.info(f"fine_tune: {fine_tune}")
    logger.info(f"adaptor: {adaptor}")
    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"weight_decay: {weight_decay}")
    logger.info(f"hidden_dropout: {hidden_dropout}")

    environment = dict()
    environment['python'] = shutil.which('python')
    environment['base_dir'] = base_dir
    environment['file_name'] = file_name
    environment['use_subset'] = use_subset

    network = dict()
    network['network_mode'] = network_mode
    network['augment'] = augment
    network['fine_tune'] = fine_tune
    network['adaptor'] = adaptor
    network['learning_rate'] = learning_rate
    network['weight_decay'] = weight_decay
    network['hidden_dropout'] = hidden_dropout

    if log_to_file is True:
        logger.info(f"log file: {os.path.join(full_log_file_path, training_log_name)}")
    else:
        logger.info(f"logging to file disabled!")

    logger.info(f"data: {dataset.columns}")


    base_logging_dir = os.path.join(base_dir, 'runs', f'run_{file_name}')
    while True:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logging_dir = f"{base_logging_dir}_{timestamp}"
        try:
            os.makedirs(logging_dir, exist_ok=False)
        except FileExistsError:
            time.sleep(500)
            continue
        break

    logger.info(f"timestamp: {timestamp}")
    logger.info(f"logging_dir: {logging_dir}")

    model, config = prepare_model_and_processor(
        model_path=model_path,
        hidden_dropout=hidden_dropout,
        fine_tune=fine_tune,
        adaptor=adaptor,
        attention=attention,
        network_mode=network_mode,
        hidden_state_policy=("last", "5"),
        attention_policy=("last", "5"),
    )

    # TODO: figure out the mechanism for saving and loading pretrained models

    # set up the dataset for use
    if use_subset_fn is not None:
        dataset = use_subset_fn(dataset)

    if augment is not None:
        augment_fn = training_augments[augment[0]]
        rng = np.random.RandomState(12345678 if len(augment) == 1 else int(augment[1]))
        augment_fn = augment_fn(rng)
    else:
        augment_fn = None

    size = (config.net_config_0.net_config.image_size, config.net_config_0.net_config.image_size)
    size_hr = None
    if config.resolution_multiplier != 1:
        size_hr = tuple(s * config.resolution_multiplier for s in size)

    if network_mode[0] == 'simple':
        trn_processor = mvds.MultiViTImageProcessor(
            size=size,
            augment=augment_fn,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
        val_processor = mvds.MultiViTImageProcessor(
            size=size,
            augment=preproc.no_augment(),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
        trn_tx_dataset, val_tx_dataset = split_dataset_v0_1(
            dataframe=dh.generate_folds_v0_1(dataset),
            data_path=images,
            cache_data=cache_data,
            processor=None,
            validation_fold=4,
            trn_processor=trn_processor,
            val_processor=val_processor,
        )
    else:
        trn_processor = mvds.MultiViTImageProcessor(
            size=size,
            size_hr=size_hr,
            augment=augment_fn,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
        val_processor = mvds.MultiViTImageProcessor(
            size=size,
            size_hr=size_hr,
            augment=preproc.no_augment(),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
        trn_tx_dataset, val_tx_dataset = multi_split_dataset_v0_1(
            dataframe=dh.generate_folds_v0_1(dataset),
            data_path=images,
            cache_data=cache_data,
            processor=None,
            validation_fold=4,
            images_per_sample=1 if network_mode[0] == 'simple' else int(network_mode[1]),
            trn_processor=trn_processor,
            val_processor=val_processor,
        )


    label_names = ('pasi', 'physician_category')

    batch_size = 16
    training_data_batch_count = len(trn_tx_dataset) // batch_size
    if len(trn_tx_dataset) % batch_size != 0:
        training_data_batch_count += 1

    training_args = TrainingArguments(
        logging_dir=os.path.join(logging_dir, "logging"),
        output_dir=os.path.join(logging_dir, "output"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=300,
        fp16=True,
        save_steps=training_data_batch_count,
        eval_steps=training_data_batch_count,
        logging_steps=1,
        learning_rate=learning_rate, #0.0001 / 4,
        save_total_limit=8,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=False,
        disable_tqdm=True,
        report_to="none",
        weight_decay=weight_decay,
        label_names=label_names,
    )

    trainer = LoggingTrainer(
        validation_save_path=os.path.join(logging_dir, "logging"),
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=None,
        train_dataset=trn_tx_dataset,
        eval_dataset=val_tx_dataset,
    )

    trainer.train()

    # save any final outstanding validation values
    trainer.save_validation_metrics()



adaptor_options = set(('probe', 'mlp'))



fine_tuning_options = set(('all', 'last', 'none', 'peft'))



training_functions = {
    '0.1': train_v0_1,
}



dataset_filter_functions = {
    'consistent_clinical': dh.use_consistent_clinical_images_v0_1,
    'all_clinical': dh.use_all_clinical_images_v0_1,
    'all_selfies': dh.use_all_selfie_images_v0_1,
    'all': None,
}



training_augments = {
    'r_crop': preproc.augument_crop_v0_1,
    'aug1': preproc.augment_light_v0_1,
}



targets = {
    'simple_pasi': None
}
