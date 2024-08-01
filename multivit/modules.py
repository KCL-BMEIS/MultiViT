from __future__ import annotations

from typing import Any, Callable, Hashable, Mapping, Sequence, Tuple

from types import SimpleNamespace

import logging

import numpy as np

import torch
from torch import nn

from transformers import ViTConfig, ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput

from multivit import attention
from multivit.config import MultiViTComponentConfig, MultiViTConfig
from multivit.utils import log_sequences_of_things



class MultiLayerPerceptron(nn.Module):


    def __init__(
            self,
            input_channels,
            hidden_channels,
            hidden_layers,
            output_channels,
            dropout=0
    ):
        super().__init__()
        if hidden_layers != 1:
            raise NotImplementedError(
                f"'hidden_layers' can only be 1 at present ('hidden_layers' is {hidden_layers})"
            )

        self.fc0 = nn.Linear(input_channels, hidden_channels)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(hidden_channels, output_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    def forward(self, t):
        t = self.fc0(t)
        t = self.act(t)
        t = self.fc1(t)
        t = self.dropout(t)
        return t


class MappingGenerator:
    """
    Generate a mapping from low-res tokens to high-res tokens for a given
    image_count and multiplier by using token_imporantance to rank the tokens.
    """

    def __init__(
            self,
            image_rows,
            image_cols,
            budget,
            image_count,
            multiplier
    ):
        if not isinstance(budget, int):
            raise TypeError(f"'budget' (type {type(budget)}) must be of type int")
        if not isinstance(image_count, int):
            raise TypeError(f"'image_count' (type {type(image_count)}) must be of type int")
        if not isinstance(multiplier, int):
            raise TypeError(f"'multiplier' (type {type(multiplier)}) must be of type int")

        self.budget = budget
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.image_count = image_count
        self.multiplier = multiplier


    def map_entries_to_high_res(self, entries):
        dst_image_cols = self.image_cols * self.multiplier
        dst_batch_entries = list()
        for b in range(len(entries)):
            dst_entries = list()
            i = 0
            for i, e in enumerate(entries[b]):
                src_t = e[0][1]
                src_y = src_t // self.image_cols
                src_x = src_t % self.image_cols
                dest_y = src_y * self.multiplier
                dest_x = src_x * self.multiplier
                for ii, (y, x) in enumerate(np.ndindex((self.multiplier, self.multiplier))):
                    dest_t = (dest_y + y) * dst_image_cols + (dest_x + x)
                    dst_entries.append([(e[0][0], dest_t), [i * self.multiplier**2 + ii]])
            dst_batch_entries.append(dst_entries)
        return dst_batch_entries


    def __call__(self, token_importances):
        """
        Generate the mapping given this set of token importances.

        `token_importances` is a list of tensors, each of which has shape (batch, num_tokens). The subset of tokens for a given batch entry (b) are
        considered together and a mapping is generated for each batch entry (b) across the list entries.

        After calculating the mappings for each batch entry, we append them back together so that we return a list
        of mappings that [batch_size][token_count] in length.
        """
        if len(token_importances) != self.image_count:
            raise ValueError(f"'token_importances' (length {len(token_importances)}) must have length {self.image_count}")

        batched_mappings = []
        for b in range(token_importances[0].shape[0]):
            all_token_importances = torch.stack([t[b] for t in token_importances], dim=0)
            indices = torch.argsort(all_token_importances.flatten(), descending=True)
            tokens = indices % all_token_importances.shape[1]
            images = indices // all_token_importances.shape[1]

            with torch.no_grad():
                ordered_indices = [(images[i].item(), tokens[i].item()) for i in range(self.budget)]

                # the mapping is simply the given input token index (each value in ordered_indices)
                # with an increasing destination token index, implemented as a generator
                mapping = list()
                for i, v in enumerate(ordered_indices):
                    mapping.append((v, list(range(i * self.multiplier**2, (i + 1) * self.multiplier**2))))
            batched_mappings.append(mapping)

            batched_mappings_hr = batched_mappings
            if self.multiplier > 1:
                batched_mappings_hr = self.map_entries_to_high_res(batched_mappings)

        return batched_mappings, batched_mappings_hr


class MultiViTPatchMapper(nn.Module):
    """
    Map patches from source images to a destination image. Depending on the overall
    network mode, this can be low res or high res images, but the mapping already
    takes that into account. Setting the multiplier to greater than 1 indicates that source
    images are 'm' times larger than destination images.

    Args:
        patch_size: the size of each patch in pixels
        dst_patch_rows: the number of patches in the row dimension
        dst_patch_cols: the number of patches in the col dimension
        multiplier: integer indicating the scale multiplier between low-res and high-res
        images. If multiplier is greater than 1, the src images will be multiplier x the
        size of the destination image
    """

    def __init__(self, patch_size, dst_patch_rows, dst_patch_cols, multiplier):
        super().__init__()
        self.patch_size = patch_size
        self.dst_patch_rows = dst_patch_rows
        self.dst_patch_cols = dst_patch_cols
        self.multiplier = multiplier
        self.src_patch_rows = dst_patch_rows * multiplier
        self.src_patch_cols = dst_patch_cols * multiplier


    def forward(self, src_images, mappings):
        dst_image = torch.zeros(
            (len(mappings), 3, self.patch_size * self.dst_patch_rows, self.patch_size * self.dst_patch_cols),
            dtype=src_images[0].dtype,
            device=src_images[0].device
        )
        # mappings are per-batch
        for b in range(len(mappings)):
            for m in mappings[b]:
                src_i = m[0][0]
                src_p = m[0][1]
                src_y = src_p // self.src_patch_cols
                src_x = src_p % self.src_patch_cols
                src_y_0 = src_y * self.patch_size
                src_y_1 = src_y_0 + self.patch_size
                src_x_0 = src_x * self.patch_size
                src_x_1 = src_x_0 + self.patch_size
                for dst_p in m[1]:
                    dst_y = dst_p // self.dst_patch_cols
                    dst_x = dst_p % self.dst_patch_cols
                    dst_y_0 = dst_y * self.patch_size
                    dst_y_1 = dst_y_0 + self.patch_size
                    dst_x_0 = dst_x * self.patch_size
                    dst_x_1 = dst_x_0 + self.patch_size
                    dst_image[b, :, dst_y_0:dst_y_1, dst_x_0: dst_x_1] =\
                        src_images[src_i, b, :, src_y_0:src_y_1, src_x_0:src_x_1]

        return dst_image



class MultiViTTokenMapper(nn.Module):
    """
    Merge image tokens with new image tokens from high-res data using the following
    mechanism:
     - Concat the low_res token and high_res_token into a single token
     - Linear concatted tensor down to original token size

    This module works with the combination of the following modes:
     - multi_input: tokens from multiple images to a single destination image
     - high_res: tokens from low_res being combined with high_res tokens

    Low res images are expected to arrive in sets of images (as lists).
    Within an image set, the dimensions are (batch, channels(rgb), height, width)

    The mapping determines for a given batch entry, which low-res token should match to which
    high-res tokens

    Example:
    ```
    [
        (0, [(0, 0), (0, 1, 2, 3)), ((1, 8), (4, 5, 6, 7)), ((1, 9), (8, 9, 10, 11), ((1, 10), (12, 13, 14, 15))]),
        (1, ...)
    ]
     - Map low-res image 0 token 0 to high-res image tokens 0, 1, 2, and 3 for sample 0
     - Map low-res image 1 token 8 to high-res image tokens 4, 5, 6, and 7 for sample 0
    ```
    """

    def __init__(self, image_count, hidden_size):
        super().__init__()
        self.fc = nn.Linear(image_count * hidden_size, hidden_size)

    def forward(self, class_tokens, patch_tokens, mapping):
        """
        Join up a set of high_res tokens freshly embedded from image-space with their
        corresponding low_res tokens.
        Args:
            tokens: A list of tokens from a vision transformer, with one tensor for each image.
                    Each tensor is expected to have the following shape: [batch_size, num_tokens, token_count]
            mapping: e.g. [(0, 0), (0, 1, 2, 3)), ((1, 8), (4, 5, 6, 7)), ((1, 9), (8, 9, 10, 11), ((1, 10), (12, 13, 14, 15))]
        """

        batch_patch_output_tokens = []
        for b in range(len(mapping)):
            output_patch_tokens = []
            for src, dest in mapping[b]:
                for d in dest:
                    output_patch_tokens.append(patch_tokens[src[0]][b, src[1], :])
            output = torch.stack(output_patch_tokens, dim=0)
            batch_patch_output_tokens.append(output)
        batch_patch_output = torch.stack(batch_patch_output_tokens, dim=0)

        # This does not appear to be necessary in the end, as the class tokens are
        # separately created and concatenated with the patch tokens in VitModel

        batch_class_output_tokens = []
        for b in range(len(mapping)):
            output_class_tokens = list()
            for s in range(len(class_tokens)):
                output_class_tokens.append(class_tokens[s][b:b+1, :, :])
            output_class_token = torch.cat(output_class_tokens, dim=2)
            batch_class_output_tokens.append(output_class_token)
        batch_class_output = torch.cat(batch_class_output_tokens, dim=0)
        batch_class_output = self.fc(batch_class_output)
        batch_output = torch.cat((batch_class_output, batch_patch_output), dim=1)
        return batch_output



class MultiViTTokenConcatenator(nn.Module):
    """
    Merge image tokens with new image tokens from high-res data using the following
    mechanism:
     - Concat the low_res token and high_res_token into a single token
     - Linear concatted tensor down to original token size

    This module works with the combination of the following modes:
     - multi_input: tokens from multiple images to a single destination image
     - high_res: tokens from low_res being combined with high_res tokens

    Low res images are expected to arrive in sets of images (as lists).
    Within an image set, the dimensions are (batch, channels(rgb), height, width)

    The mapping determines for a given batch entry, which low-res token should match to which
    high-res tokens

    Example:
    ```
    [
        (0, [(0, 0), (0, 1, 2, 3)), ((1, 8), (4, 5, 6, 7)), ((1, 9), (8, 9, 10, 11), ((1, 10), (12, 13, 14, 15))]),
        (1, ...)
    ]
     - Map low-res image 0 token 0 to high-res image tokens 0, 1, 2, and 3 for sample 0
     - Map low-res image 1 token 8 to high-res image tokens 4, 5, 6, and 7 for sample 0
    ```
    """

    def __init__(self, token_size):
        super().__init__()
        self.fc = nn.Linear(token_size * 2, token_size)

    def forward(self, low_res, high_res):
        """
        Join up a set of high_res tokens freshly embedded from image-space with their
        corresponding low_res tokens.
        Args:
            low_res: The tokens from a vision transformer, expected to have the following
                     shape: [batch_size, num_tokens, token_count]
            high_res: The tokens from the preliminary embedding layer of a vision transformer,
                     expected to have the following shape[batch_size, num_tokens, token_size]
            mapping: e.g. [(0, 0), (0, 1, 2, 3)), ((1, 8), (4, 5, 6, 7)), ((1, 9), (8, 9, 10, 11), ((1, 10), (12, 13, 14, 15))]
        """
        if not all(a == b for a, b in zip(low_res.shape, high_res.shape)):
            raise ValueError(f"'low_res' (shape {low_res.shape}) and 'high_res' (shape {high_res.shape}) must have the same shape")

        return self.fc(torch.cat((low_res, high_res), dim=2))




class MultiViTFakeEmbedding(nn.Module):
    """
    The purpose of this class is solely to avoid subclassing ViTModel. It replaces the
    embeddings module of a ViTModel instance with module that is essentially a no-op
    module, but with the extra properties required to satisfy ViTModel.
    """

    def __init__(self, hidden_size, dtype: torch.dtype = torch.float32):
        super().__init__()
        weight = SimpleNamespace(dtype=dtype)
        projection = SimpleNamespace(weight=weight)
        patch_embeddings=SimpleNamespace(projection=projection)
        self.patch_embeddings = patch_embeddings

    def forward(self, pixel_values, **kwargs):
        return pixel_values



class MultiViTEncoder(nn.Module):

    def __init__(
            self,
            vit_encoder,
            output_first_hidden_state: bool,
            hidden_state_selector: Callable | None = None,
            attention_selector: Callable | None = None
    ):
        super().__init__()
        self.layer = vit_encoder.layer
        self.gradient_checkpointing = vit_encoder.gradient_checkpointing
        self.output_first_hidden_state = output_first_hidden_state
        self.hidden_state_selector = hidden_state_selector
        self.attention_selector = attention_selector


    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: torch.Tensor | None = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True
    ) -> tuple | BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            cur_hidden_states = hidden_states if self.output_first_hidden_state else None
            all_hidden_states = all_hidden_states + (cur_hidden_states,)

        for i, layer_module in enumerate(self.layer):

            layer_head_mask = head_mask[i] if head_mask is not None else None

            cur_output_attentions = self.attention_selector(i)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    cur_output_attentions
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, cur_output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                cur_attentions = layer_outputs[1] if cur_output_attentions else None
                all_self_attentions = all_self_attentions + (cur_attentions,)

            if output_hidden_states:
                cur_hidden_states = hidden_states if self.hidden_state_selector(i) else None
                all_hidden_states = all_hidden_states + (cur_hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )



def task_head_from_adaptor(adaptor_desc: dict):
    if not isinstance(adaptor_desc, dict):
        raise TypeError(f"'adaptor_desc' (type {type(adaptor_desc)}) must be of type dict")

    if 'head' not in adaptor_desc:
        raise KeyError(f"'adaptor_desc' (with keys {list(adaptor_desc.keys())}) must contain a 'head' key")

    adaptor_heads = ('linear', 'mlp')
    adaptor_head = adaptor_desc['head']

    if adaptor_head['head'] == 'linear':
        head = nn.Linear(**adaptor_head['kwargs'])
    elif adaptor_head['head'] == 'mlp':
        head = MultiLayerPerceptron(
            **adaptor_head['kwargs'],
        )
    else:
        raise ValueError(f"'adaptor_desc'['head'] ({adaptor_desc['head']}) must be one of {adaptor_heads}")

    return head


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        labels = labels.squeeze(dim=1)
        loss_out = self.loss(logits, labels)
        loss_out = loss_out.unsqueeze(dim=1)
        return loss_out


def task_loss_from_adaptor(adaptor_desc: dict):
    if not isinstance(adaptor_desc, dict):
        raise TypeError(f"'adaptor_desc' (type {type(adaptor_desc)}) must be of type dict")

    if 'loss' not in adaptor_desc:
        raise KeyError(f"'adaptor_desc' (with keys {adaptor_desc.keys()}) must contain a 'loss' key")

    adaptor_losses = ('mse', 'xent')

    adaptor_loss = adaptor_desc['loss']
    if adaptor_loss['loss'] == 'mse':
        loss = nn.MSELoss(reduction='none')
    elif adaptor_loss['loss'] == 'xent':
        loss = CrossEntropyLoss()
    else:
        raise ValueError(f"'adaptor_desc'['loss']['loss'] ({adaptor_loss['loss']}) must be one of {adaptor_losses}")

    return loss


class MultiViTHead(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            adaptors: Any | None = None,
    ):
        logger = logging.getLogger(f"{__class__.__name__}.__init__")
        super().__init__()

        task_heads = dict()
        task_losses = dict()
        for adaptor in adaptors or []:
            task_head = task_head_from_adaptor(adaptor)
            task_loss = task_loss_from_adaptor(adaptor)
            task_heads[adaptor['loss']['label']] = task_head
            task_losses[adaptor['loss']['label']] = task_loss

        self.task_heads = nn.ModuleDict(task_heads)
        self.task_losses = nn.ModuleDict(task_losses)



    def forward(
            self,
            pixel_values: torch.Tensor | None = None,
            labels: Mapping[Hashable, torch.Tensor] | None = None,
    ):
        logger = logging.getLogger(f"{__class__.__name__}.{self.forward.__name__}")

        labels = {k: v.to(pixel_values.device) for k, v in labels.items()}

        overall_loss = None
        all_logits = dict()
        all_losses = dict()
        for k, v in self.task_heads.items():
            outputs = v(pixel_values)
            logits = outputs[:, 0, :]

            losses = self.task_losses[k](logits, labels[k])
            all_losses[k] = losses
            all_logits[k] = logits
            if overall_loss is None:
                overall_loss = losses.mean()
            else:
                overall_loss += losses.mean()

        return {'loss': overall_loss, 'logits': all_logits, 'losses': all_losses}



class MultiViTComponent(ViTPreTrainedModel):

    def __init__(
        self,
        config: MultiViTComponentConfig,
    ) -> None:
        """
        Modes:
         - stage 0:
           - no need to modify the ViTModel; leave everything as is
         - stage 1:
           - extract the ViTModel embedder to use here
           - replace the ViTModel embedder with a fake embedder that passes whatever it is
             given to the encoder without modifying it
           - mode == "multi":
             - 'pixel_values' is the composite image data
             - 'token_values' is None
             - put pixel_values through the extracted embedder
             - pass the tokens to the VitModel as 'pixel_values'
           - mode == "full":
             - resolution_multiplier == 1:
               - 'pixel_values' is None
               - 'token_values' is the composite token data
               - pass the tokens to the VitModel as 'pixel_values'
             - resolution_multiplier > 1:
               - 'pixel_values' is the composite image data
               - 'token_values' is the composite token data
               - pass the pixel_values through the extracted embedder
               - concatenate tokenised pixel_values with token_values
               - pass the resulting tokens to the ViTModel as 'pixel_values'
        """
        super().__init__(config)
        logger = logging.getLogger(f"{__class__.__name__}.__init__")

        if config.stage not in (0, 1):
            raise ValueError(f"'stage' ({config.stage}) must be one of (0, 1)")

        if config.stage == 0:
            if config.concat_embeddings == True:
                raise ValueError(f"'config.concat_embeddings' ({config.concat_embeddings}) must be False "
                                 f" if 'config.stage' ({config.stage}) is 0")

        task_head = MultiViTHead(config.net_config.hidden_size, config.adaptors)

        self.vit = ViTModel.from_pretrained(
            config.weights_name, config=config.net_config, add_pooling_layer=False, local_files_only=True,
        )
        self.vit.encoder = MultiViTEncoder(
            self.vit.encoder,
            False,
            lambda i: config.output_hidden_states_by_layer[i],
            lambda i: config.output_attentions_by_layer[i],
        )


        # TODO: the task heads should do the same thing as the default pooler does, but should there be a pooler
        # per task?

        # TODO: None of this is nice. You should probably subclass ViTModel rather than chopping off / replacing the
        # embeddings layer.
        self.embeddings = None
        if config.stage == 1:
            embeddings = self.vit.embeddings
            self.vit.embeddings = MultiViTFakeEmbedding(
                config.net_config.hidden_size,
                dtype=embeddings.patch_embeddings.projection.weight.dtype
            )
            if config.use_embeddings:
                self.embeddings = embeddings
            else:
                self.cls_token = nn.Parameter(torch.randn(1, 1, config.net_config.hidden_size))


            self.concatenator = None
            if config.concat_embeddings is True:
                self.concatenator = MultiViTTokenConcatenator(config.net_config.hidden_size)

        self.task_head = nn.Identity() if task_head is None else task_head


    def forward(
        self,
        pixel_values: torch.Tensor,
        token_values: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        labels: Mapping[str, torch.Tensor] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
        return_dict: bool | None = None,
    ) -> dict:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logger = logging.getLogger(f"{__class__.__name__}.{self.forward.__name__}")

        output_hidden_states_ = output_hidden_states or self.config.output_hidden_states
        output_attentions_ = output_attentions or self.config.output_attentions

        if self.embeddings is not None:
            if self.concatenator is not None:
                token_values = self.concatenator(token_values, self.embeddings(pixel_values))
            else:
                token_values = self.embeddings(pixel_values)
        else:
            token_values = token_values if pixel_values is None else pixel_values

        output = self.vit(
            token_values, # if this is a stage 1 network, we always pass the token values
            head_mask=head_mask,
            output_attentions=output_attentions_,
            output_hidden_states=output_hidden_states_,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )
        sequence_output = output[0]

        result = self.task_head(sequence_output, labels)
        if output.attentions is not None:
            result['attentions'] = output.attentions
        if output.hidden_states is not None:
            result['hidden_states'] = output.last_hidden_state
        return result


    def _get_class_tokens(self, outputs):
        return outputs[:, 0, :]



class MultiViT(ViTPreTrainedModel):


    def __init__(
        self,
        config: MultiViTConfig | None = None,
    ):
        """
        Configurations:
         - Two stage, no back prop between stages
        """
        super().__init__(config)

        logger = logging.getLogger(f"{__class__.__name__}.__init__")

        network_mode_ = config.network_mode

        self.mapping_generator = None
        self.token_mapper = None
        self.patch_mapper = None

        if network_mode_ == "simple":
            model_0 = MultiViTComponent(config.net_config_0)
            self.model_0 = model_0
            self.model_1 = None

        else:
            model_0 = MultiViTComponent(config.net_config_0)
            self.model_0 = model_0

            # NOTE: when the resolution multiplier is greater than 1, we need
            # to reduce the number of tokens we are mapping from the source tokens,
            # as we are limited by the number of detailed tokens we can get from
            # the high-res images
            tokens_per_dim = config.net_config_0.net_config.image_size // config.net_config_0.net_config.patch_size
            token_count = tokens_per_dim**2 // config.resolution_multiplier**2

            self.attention_weights = attention.ViTAttentionSimple()

            self.mapping_generator = MappingGenerator(
                image_rows=tokens_per_dim,
                image_cols=tokens_per_dim,
                budget=token_count,
                image_count=config.image_count,
                multiplier=config.resolution_multiplier
            )

            if network_mode_ == "full":
                self.token_mapper = MultiViTTokenMapper(
                    image_count=config.image_count,
                    hidden_size=config.net_config_0.net_config.hidden_size
                )

            if network_mode_ == "multi" or config.resolution_multiplier > 1:
                patch_size = config.net_config_0.net_config.patch_size
                image_size = config.net_config_0.net_config.image_size
                self.patch_mapper = MultiViTPatchMapper(
                    patch_size= patch_size,
                    dst_patch_rows = image_size // patch_size,
                    dst_patch_cols = image_size // patch_size,
                    multiplier=config.resolution_multiplier
                )

            model_1 = MultiViTComponent(config.net_config_1)
            self.model_1 = model_1

        self.network_mode = network_mode_

        self.config = config



    def forward(
            self,
            pixel_values: torch.Tensor = None,
            pixel_values_hr: torch.Tensor | None = None,
            head_mask: torch.Tensor | None = None,
            labels: Mapping[str, torch.Tensor] | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            interpolate_pos_encoding: bool | None = None,
            return_dict: bool | None = None,
    ):

        logger = logging.getLogger(f"{__class__.__name__}.{self.forward.__name__}")
        logger.info(f"image_shape: {pixel_values.shape}")
        if self.network_mode == "simple":
            result = self.model_0(
                pixel_values=pixel_values,
                head_mask=head_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            return {
                'loss_0': result['loss'],
                'output_0': result,
            }
        else:
            if pixel_values_hr is None:
                if self.config.resolution_multiplier > 1:
                    raise ValueError(f"'pixel_values_hr' must be provided by the dataset if "
                                     "'resolution_multiplier' is greater than 1")
                else:
                    pixel_values_hr = pixel_values
            # TODO: this is for inference rather than training
            outputs_0 = list()
            losses_0 = list()
            for i in range(self.config.image_count):
                result_0 = self.model_0(
                    pixel_values=pixel_values[i, ...].squeeze(dim=1),
                    head_mask=head_mask,
                    labels={k: v[i, ...] for k, v in labels.items()},
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    interpolate_pos_encoding=interpolate_pos_encoding,
                    return_dict=return_dict,
                )
                losses_0.append(result_0['loss'])
                outputs_0.append(result_0)

            # TODO: should the class mapping also be passed to the second network in the 'full' network modes?

            attentions = [o['attentions'] for o in outputs_0]
            class_tokens = [o['hidden_states'][:, 0:1, :] for o in outputs_0]
            patch_tokens = [o['hidden_states'][:, 1:, :] for o in outputs_0]
            token_weights = [self.attention_weights(a) for a in attentions]
            token_size = self.config.net_config_1.net_config.patch_size
            composite_image_shape = (
                3,
                self.config.net_config_1.net_config.image_size,
                self.config.net_config_1.net_config.image_size
            )
            mapping, mapping_hr = self.mapping_generator(token_weights)

            if self.token_mapper is not None:
                composite_tokens = self.token_mapper(class_tokens, patch_tokens, mapping)
            else:
                composite_tokens = None

            if self.patch_mapper is not None:
                composite_image = self.patch_mapper(pixel_values_hr, mapping_hr)
            else:
                composite_image = None

            composite_labels = {k: v[0, ...] for k, v in labels.items() if k != 'iid'}
            result_1 = self.model_1(
                pixel_values=composite_image,
                token_values=composite_tokens,
                head_mask=head_mask,
                labels=composite_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            result_dict = {
                "losses_0": losses_0,
                "outputs_0": outputs_0,
                "loss_1": result_1['loss'],
                "output_1": result_1,
            }
            if mapping is not None:
                result_dict['mappings'] = mapping
            if mapping_hr is not None:
                result_dict['mappings_hr'] = mapping_hr
            return result_dict


def get_freeze_function(mode):
    if isinstance(mode, str):
        mode = tuple(mode)
    if isinstance(mode, (tuple, list)):

        if mode[0] == 'probe':
            return freeze_all_fn()
        elif mode[0] == 'none':
            return freeze_none_fn()
        elif mode[0] == 'last':
            if len(mode) != 2:
                raise ValueError(f"if 'mode' is 'last' the number of layers to be frozen must be specified ('mode' is {mode}")
            return freeze_last_n_fn(mode[1])
        elif mode[0] == 'peft':
            raise NotImplementedError()
        else:
            raise ValueError(f"'mode' must be one of 'probe' 'none', 'last', or 'peft' but is {mode[0]}")


def freeze_all_fn():
    def _inner(module):
        if not isinstance(module, nn.Module):
            raise TypeError(f"'module' must be of type torch.nn.Module but is of type {type(module)}")

        for p in module.parameters():
            p.require_grads = False

        return module
    return _inner



def freeze_last_n_fn(n):
    def _inner(module):
        if not isinstance(module, nn.Module):
            raise TypeError(f"'module' must be of type torch.nn.Module but is of type {type(module)}")

        max_layer_index = None
        for np, p in module.named_parameters():
            if 'vit.encoder.layer' in np:
                layer_index = int(np.split('.')[3])
                if max_layer_index is None or layer_index > max_layer_index:
                    max_layer_index = layer_index
        first_unfrozen_index = max_layer_index - int(n) + 1
        # pass 2: freeze all layers with a lower index than the first unfrozen index
        for np, p in module.named_parameters():
            if 'vit.encoder.layer' in np and int(np.split('.')[3]) >= first_unfrozen_index:
                p.requires_grad = False

        return module
    return _inner



def freeze_none_fn():
    def _inner(module):
        if not isinstance(module, nn.Module):
            raise TypeError(f"'module' must be of type torch.nn.Module but is of type {type(module)}")
        return module
    return _inner



network_modes = ('simple', 'multi', 'full')



def parse_network_mode(
        network_mode
):
    """
    Expected:
     - "simple": single element
     - "multi": three elements '("multi", <image_count>, <resolution_multiplier>)
     - "full": three elements '("multi", <image_count>, <resolution_multiplier>)
    """
    if network_mode[0] not in network_modes:
        raise ValueError(f"'network_mode' ({network_mode[0]}) must be one of {network_modes}")
    image_count = None
    resolution_multiplier = 1
    if network_mode[0] == "simple":
        if len(network_mode) != 1:
            raise ValueError(f"'network_mode' ({network_mode}): 'simple' should have no additional elements")
    elif network_mode[0] == 'multi' or network_mode[0] == 'full':
        if len(network_mode) != 3:
            raise ValueError(f"'network_mode' ({network_mode}): '{network_mode[0]}' should have two additional "
                             "elements for image_count and resolution_multiplier")
        image_count = int(network_mode[1])
        resolution_multiplier = int(network_mode[2])
    return network_mode[0], image_count, resolution_multiplier



def get_per_layer_output_policy(policy, num_hidden_layers):
    if isinstance(policy, str):
        policy = tuple(policy)

    if policy[0] == 'all':
        return [True for _ in range(num_hidden_layers)]
    elif policy[0] == 'none':
        return [False for _ in range(num_hidden_layers)]
    elif policy[0] == 'first':
        threshold = int(policy[1])
        return [i < threshold for i in range(num_hidden_layers)]
    elif policy[0] == 'last':
        threshold = int(policy[1])
        return [i >= (num_hidden_layers - threshold) for i in range(num_hidden_layers)]
    else:
        raise ValueError(f"The first element of 'policy' {policy} must be one of 'all', 'none', 'first', or 'last'")



def prepare_adaptor(
        config,
        logger=None
):
    if config['adaptor'] == 'linear':
        module = nn.Linear(config['input_size'], config['output_size'])
    elif config['adaptor'] == 'mlp':
        module = MultiLayerPerceptron(
            input_channels=config['input_size'],
            hidden_channels=config['hidden_size'],
            hidden_layers=config['hidden_layers'],
            output_channels=config['output_size'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"config['adaptor' ({config['adaptor']}) must be one of ('linear', 'mlp')")

    return module



def task_loss_from_config(
        task_config: dict | callable,
        logger=None
):
    if task_config['loss'] == 'xent':
        loss = nn.CrossEntropyLoss(reduction='none')
    elif task_config['loss'] == 'mse':
        loss = nn.MSELoss(reduction='none')
    return loss
