from __future__ import annotations

from typing import Callable, Sequence

import logging

import torch
from torch import nn

from transformers import PretrainedConfig
from transformers import ViTConfig



class MultiViTConfig(PretrainedConfig):

    model_type = "multivit"


    def __init__(
        self,
        network_mode: str | None = None,
        image_count: int | None = None,
        resolution_multiplier: int | None = None,
        net_config_0: MultiViTComponentConfig | None = None,
        net_config_1: MultiViTComponentConfig | None = None,
        **kwargs
    ):
        """
        Args:
            network_mode: One of "solo", "dual_0", "dual_1", or "dual_2"
            image_count: If set, the first network will be called multiple times with a different image from the
                         same sample. These will be concatenated together to a composite image during the mapping
                         step.
            net_config_0: The configuration for the first network. This must always be set
            net_config_1: The configuration for the second network. This is only used if network_mode is one of the
                          "dual_*" modes
        """
        super().__init__(**kwargs)
        self.network_mode = network_mode
        self.resolution_multiplier = resolution_multiplier
        self.image_count = image_count
        self.net_config_0 = net_config_0
        self.net_config_1 = net_config_1


class MultiViTComponentConfig(PretrainedConfig):

    model_type = "multivit_component"


    def __init__(
        self,
        net_config=None,
        weights_name=None,
        stage: int | None = None,
        use_embeddings: bool | None = None,
        concat_embeddings: bool | None = None,
        adaptors: Sequence[Sequence[str] | Callable] | None = None,
        fine_tuning: Sequence[str] = ("none"),
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states_by_layer: Callable | None = None,
        output_attentions_by_layer: Callable | None = None,
        **kwargs,
    ):
        """
        Args:
            net_config: ViTConfig
            weights_name: str
            supress_embeddings: if this is set, this component is being passed tokens rather than image patches and so
                shouldn't call ViTModel.embeddings
            concat_embeddings: if this is set, this component should embed high-res image tokens and then concatentate
                them with the low-res image tokens before passing them to self.vit
            # classifier_heads: nn.Module
            # classifier_loss: Callable
            # regressor_heads: nn.Module
            # regressor_loss: Callable
            adaptors: Sequence[Sequence[str] | Callable]: adaptors for the network
            fine_tuning: Sequence[str]
            adaptor: Sequence[str]
            output_hidden_states: bool: if this is set, the model will enable overall hidden states outputting.
            output_attentions: bool: if this is set, the model will enable overall attention outputting.
            output_hidden_states_by_layer: Callable: if this is set and output_hidden_states is True, this controls
                whether hidden states are returned for a given layer
            output_attentions_by_layer: Callable: if this is set and output_attentions is True, this controls whether
                attentions are returned for a given layer
        """
        super().__init__(**kwargs)
        self.net_config = ViTConfig() if net_config is None else net_config
        self.weights_name = weights_name
        self.stage = stage
        self.use_embeddings = use_embeddings
        self.concat_embeddings = concat_embeddings
        self.adaptors = adaptors
        self.fine_tuning = fine_tuning
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.output_hidden_states_by_layer = output_hidden_states_by_layer
        self.output_attentions_by_layer = output_attentions_by_layer
