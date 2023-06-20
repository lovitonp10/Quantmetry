import logging
import random
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from torch import nn
from transformers import InformerPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    SampleTSPredictionOutput,
    Seq2SeqTSModelOutput,
    Seq2SeqTSPredictionOutput,
)
from transformers.models.informer.modeling_informer import (
    InformerConvLayer,
    InformerDecoderLayer,
    InformerEncoderLayer,
    InformerFeatureEmbedder,
    InformerMeanScaler,
    InformerNOPScaler,
    InformerSinusoidalPositionalEmbedding,
    InformerStdScaler,
    InformerValueEmbedding,
    _expand_mask,
    _make_causal_mask,
)
from transformers.time_series_utils import (
    NegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from utils.utils_informer.configuration_informer import CustomInformerConfig

logger = logging.getLogger(__name__)

"""
Description: This file is a modified version of modeling_informer.py
that allows Informer model to support past features.
Original File:
https://github.com/huggingface/transformers/tree/main/src/transformers/models/informer/modeling_informer.py
"""


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


def weighted_average(
    input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`,
    masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(
            weights != 0, input_tensor * weights, torch.zeros_like(input_tensor)
        )
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


class CustomInformerModel(InformerPreTrainedModel):
    def __init__(self, config: CustomInformerConfig):
        super().__init__(config)

        if config.scaling == "mean" or config.scaling:
            self.scaler = InformerMeanScaler(dim=1, keepdim=True)
        elif config.scaling == "std":
            self.scaler = InformerStdScaler(dim=1, keepdim=True)
        else:
            self.scaler = InformerNOPScaler(dim=1, keepdim=True)

        if config.num_static_categorical_features > 0:
            self.embedder = InformerFeatureEmbedder(
                cardinalities=config.cardinality,
                embedding_dims=config.embedding_dimension,
            )

        # transformer encoder-decoder and mask initializer
        self.encoder = CustomInformerEncoder(config)
        self.decoder = CustomInformerDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def _past_length(self) -> int:
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence. Returns a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].

        Args:
            sequence: Tensor
                The sequence from which lagged subsequences should be extracted. Shape: (N, T, C).
            subsequences_length : int
                Length of the subsequences to be extracted.
            shift: int
                Shift the lags by this amount back.
        """
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.config.lags_sequence]

        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(
                f"lags cannot go further than history length, found lag {max(indices)} "
                f"while history length is only {sequence_length}"
            )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def create_network_inputs(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        past_dynamic_real_features: Optional[torch.Tensor] = None,
    ):
        accelerator = Accelerator()
        device = accelerator.device

        # time feature
        time_feat = (
            torch.cat(
                (
                    past_time_features[:, self._past_length - self.config.context_length :, ...],
                    future_time_features,
                ),
                dim=1,
            )
            if future_values is not None
            else past_time_features[:, self._past_length - self.config.context_length :, ...]
        )

        # past dynamic real features
        if past_dynamic_real_features is not None:

            time_feat = (
                torch.cat(
                    (
                        time_feat,
                        past_dynamic_real_features[
                            :, self._past_length - self.config.context_length :, ...
                        ],
                    ),
                    dim=-1,
                )
                if future_values is None
                else torch.cat(
                    (
                        time_feat,
                        torch.cat(
                            (
                                past_dynamic_real_features[
                                    :, self._past_length - self.config.context_length :, ...
                                ],
                                torch.zeros(
                                    (
                                        past_dynamic_real_features.shape[0],
                                        self.config.prediction_length,
                                        past_dynamic_real_features.shape[2],
                                    )
                                ).to(device),
                            ),
                            dim=1,
                        ),
                    ),
                    dim=-1,
                )
            )

        # target
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        context = past_values[:, -self.config.context_length :]
        observed_context = past_observed_mask[:, -self.config.context_length :]
        _, loc, scale = self.scaler(context, observed_context)

        inputs = (
            (torch.cat((past_values, future_values), dim=1) - loc) / scale
            if future_values is not None
            else (past_values - loc) / scale
        )

        # static features
        log_abs_loc = (
            loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
        )
        log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

        if static_real_features is not None:
            static_feat = torch.cat((static_real_features, static_feat), dim=1)
        if static_categorical_features is not None:
            embedded_cat = self.embedder(static_categorical_features)
            static_feat = torch.cat((embedded_cat, static_feat), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

        # all features
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # lagged features
        subsequences_length = (
            self.config.context_length + self.config.prediction_length
            if future_values is not None
            else self.config.context_length
        )
        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs, subsequences_length=subsequences_length
        )
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

        if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
            raise ValueError(
                f"""input length {reshaped_lagged_sequence.shape[1]} and
                time feature lengths {time_feat.shape[1]} does not match"""
            )

        # transformer inputs
        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        return transformer_inputs, loc, scale, static_feat

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        past_dynamic_real_features: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqTSModelOutput, Tuple]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import InformerModel

        >>> file = hf_hub_download(
        ...     repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt",
        repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly")

        >>> # during training, one provides both past and future values
        >>> # as well as possible additional features
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_inputs, loc, scale, static_feat = self.create_network_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
            past_dynamic_real_features=past_dynamic_real_features,
        )

        if encoder_outputs is None:
            enc_input = transformer_inputs[:, : self.config.context_length, ...]
            encoder_outputs = self.encoder(
                inputs_embeds=enc_input,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput
        # when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        dec_input = transformer_inputs[:, self.config.context_length :, ...]
        decoder_outputs = self.decoder(
            inputs_embeds=dec_input,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs + (loc, scale, static_feat)

        return Seq2SeqTSModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            loc=loc,
            scale=scale,
            static_features=static_feat,
        )


class CustomInformerForPrediction(InformerPreTrainedModel):
    def __init__(self, config: CustomInformerConfig):
        super().__init__(config)
        self.model = CustomInformerModel(config)
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.parameter_projection = self.distribution_output.get_parameter_projection(
            self.model.config.d_model
        )
        self.target_shape = self.distribution_output.event_shape

        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # Initialize weights of distribution_output and apply final processing
        self.post_init()

    def output_params(self, dec_output):
        return self.parameter_projection(dec_output)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        past_dynamic_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqTSModelOutput, Tuple]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import InformerForPrediction

        >>> file = hf_hub_download(
        ...     repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt",
          repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly")

        >>> # during training, one provides both past and future values
        >>> # as well as possible additional features
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # during inference, one only provides past values
        >>> # as well as possible additional features
        >>> # the model autoregressively generates future values
        >>> outputs = model.generate(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> mean_prediction = outputs.sequences.mean(dim=1)
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if future_values is not None:
            use_cache = False

        outputs = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_dynamic_real_features=past_dynamic_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        prediction_loss = None
        params = None
        if future_values is not None:
            params = self.output_params(outputs[0])  # outputs.last_hidden_state
            # loc is 3rd last and scale is 2nd last output
            distribution = self.output_distribution(params, loc=outputs[-3], scale=outputs[-2])

            loss = self.loss(distribution, future_values)

            if future_observed_mask is None:
                future_observed_mask = torch.ones_like(future_values)

            if len(self.target_shape) == 0:
                loss_weights = future_observed_mask
            else:
                loss_weights, _ = future_observed_mask.min(dim=-1, keepdim=False)

            prediction_loss = weighted_average(loss, weights=loss_weights)

        if not return_dict:
            outputs = ((params,) + outputs[1:]) if params is not None else outputs[1:]
            return ((prediction_loss,) + outputs) if prediction_loss is not None else outputs

        return Seq2SeqTSPredictionOutput(
            loss=prediction_loss,
            params=params,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            loc=outputs.loc,
            scale=outputs.scale,
            static_features=outputs.static_features,
        )

    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        past_dynamic_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SampleTSPredictionOutput:
        r"""
        Greedily generate sequences of sample predictions from a model with a probability
        distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or
            `(batch_size, sequence_length, input_size)`):
                Past values of the time series, that serve as context in order to predict
                the future. The sequence size
                of this tensor must be larger than the `context_length` of the model, since
                the model will use the
                larger size to construct lag features, i.e. additional values from the past
                which are added in order to
                serve as "extra context".

                The `sequence_length` here is equal to
                `config.context_length` + `max(config.lags_sequence)`, which if
                no `lags_sequence` is configured, is equal to `config.context_length` + 7
                (as by default, the largest
                look-back index in `config.lags_sequence` is 7). The property `_past_length`
                returns the actual length
                of the past.

                The `past_values` is what the Transformer encoder gets as input
                (with optional additional features,
                such as `static_categorical_features`, `static_real_features`,
                `past_time_features` and lags).

                Optionally, missing values need to be replaced with zeros and indicated via
                  the `past_observed_mask`.

                For multivariate time series, the `input_size` > 1 dimension is required and
                corresponds to the number
                of variates in the time series per time step.
            past_time_features (`torch.FloatTensor` of shape
            `(batch_size, sequence_length, num_features)`):
                Required time features, which the model internally will add to `past_values`.
                  These could be things
                like "month of year", "day of the month", etc. encoded as vectors
                (for instance as Fourier features).
                These could also be so-called "age" features, which basically help the model know
                "at which point in
                life" a time-series is. Age features have small values for distant past time steps
                and increase
                monotonically the more we approach the current time step. Holiday features are also
                 a good example of
                time features.

                These features serve as the "positional encodings" of the inputs.
                So contrary to a model like BERT,
                where the position encodings are learned from scratch internally as
                parameters of the model, the Time
                Series Transformer requires to provide additional time features.
                The Time Series Transformer only
                learns additional embeddings for `static_categorical_features`.

                Additional dynamic real covariates can be concatenated to this tensor,
                with the caveat that these
                features must but known at prediction time.

                The `num_features` here is equal to
                `config.`num_time_features` + `config.num_dynamic_real_features`.
            future_time_features (`torch.FloatTensor` of shape
            `(batch_size, prediction_length, num_features)`):
                Required time features for the prediction window, which the model
                internally will add to sampled
                predictions. These could be things like "month of year",
                "day of the month", etc. encoded as vectors
                (for instance as Fourier features). These could also be so-called
                "age" features, which basically help
                the model know "at which point in life" a time-series is.
                Age features have small values for distant
                past time steps and increase monotonically the more we approach
                the current time step. Holiday features
                are also a good example of time features.

                These features serve as the "positional encodings" of the inputs.
                So contrary to a model like BERT,
                where the position encodings are learned from scratch internally
                as parameters of the model, the Time
                Series Transformer requires to provide additional time features.
                The Time Series Transformer only
                learns additional embeddings for `static_categorical_features`.

                Additional dynamic real covariates can be concatenated to this tensor,
                  with the caveat that these
                features must but known at prediction time.

                The `num_features` here is equal to
                `config.`num_time_features` + `config.num_dynamic_real_features`.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or
              `(batch_size, sequence_length, input_size)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing.
                Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            static_categorical_features (`torch.LongTensor` of shape
            `(batch_size, number of static categorical features)`, *optional*):
                Optional static categorical features for which the model will learn an embedding,
                  which it will add to
                the values of the time series.

                Static categorical features are features which have the same value for
                all time steps (static over time).

                A typical example of a static categorical feature is a time series ID.
            static_real_features (`torch.FloatTensor` of shape
            `(batch_size, number of static real features)`, *optional*):
                Optional static real features which the model will add to the values of
                the time series.

                Static real features are features which have the same value for all
                time steps (static over time).

                A typical example of a static real feature is promotion information.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

        Return:
            [`SampleTSPredictionOutput`] where the outputs `sequences` tensor will have shape
              `(batch_size, number of
            samples, prediction_length)` or `(batch_size, number of samples, prediction_length,
              input_size)` for
            multivariate predictions.
        """
        outputs = self(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=None,
            past_dynamic_real_features=past_dynamic_real_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=True,
        )

        accelerator = Accelerator()
        device = accelerator.device

        decoder = self.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_features.shape[1], -1
        )
        to_cat = (expanded_static_feat, future_time_features)
        if past_dynamic_real_features is not None:
            future_unknown_features = torch.zeros(
                (
                    past_dynamic_real_features.shape[0],
                    self.config.prediction_length,
                    past_dynamic_real_features.shape[2],
                )
            ).to(device)
            to_cat += (future_unknown_features,)

        features = torch.cat(to_cat, dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        future_samples = []

        # greedy decoding
        for k in range(self.config.prediction_length):
            lagged_sequence = self.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1
            )

            dec_output = decoder(
                inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden
            )
            dec_last_hidden = dec_output.last_hidden_state

            params = self.parameter_projection(dec_last_hidden[:, -1:])
            distr = self.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
            next_sample = distr.sample()

            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)

        return SampleTSPredictionOutput(
            sequences=concat_future_samples.reshape(
                (-1, num_parallel_samples, self.config.prediction_length) + self.target_shape,
            )
        )


class CustomInformerEncoder(InformerPreTrainedModel):
    """
    Informer encoder consisting of *config.encoder_layers* self attention layers with
    distillation layers. Each
    attention layer is an [`InformerEncoderLayer`].

    Args:
        config: InformerConfig
    """

    def __init__(self, config: CustomInformerConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.gradient_checkpointing = False
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = InformerValueEmbedding(
            feature_size=config.feature_size, d_model=config.d_model
        )
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        self.layers = nn.ModuleList(
            [InformerEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        if config.distil:
            self.conv_layers = nn.ModuleList(
                [InformerConvLayer(config.d_model) for _ in range(config.encoder_layers - 1)]
            )
            self.conv_layers.append(None)
        else:
            self.conv_layers = [None] * config.encoder_layers

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`,
              *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected
                  in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape
            `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass
                an embedded representation.
                This is useful if you want more control over how to convert `input_ids`
                indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states`
                under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size())

        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"""The head_mask should be specified for {len(self.layers)}
                    layers, but it is for"""
                    f" {head_mask.size()[0]}."
                )

        for idx, (encoder_layer, conv_layer) in enumerate(zip(self.layers, self.conv_layers)):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                    if conv_layer is not None:
                        output = torch.utils.checkpoint.checkpoint(conv_layer, layer_outputs[0])
                        layer_outputs = (output,) + layer_outputs[1:]
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )
                    if conv_layer is not None:
                        output = conv_layer(layer_outputs[0])
                        layer_outputs = (output,) + layer_outputs[1:]

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CustomInformerDecoder(InformerPreTrainedModel):
    """
    Informer decoder consisting of *config.decoder_layers* layers.
    Each layer is a [`InformerDecoderLayer`]

    Args:
        config: InformerConfig
    """

    def __init__(self, config: CustomInformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = InformerValueEmbedding(
            feature_size=config.feature_size, d_model=config.d_model
        )
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        self.layers = nn.ModuleList(
            [InformerDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape
            `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder.
                  Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape
            `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of
                encoder input_ids.
                Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape
            `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules.
                Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape
            `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder
                to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*,
            returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with
                each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
                and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention
                blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input)
                to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the
                last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of
                shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape
            `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an
                embedded representation.
                This is useful if you want more control over how to convert `input_ids`
                indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = inputs_embeds.size()[:-1]

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(
            inputs_embeds.size(), past_key_values_length=self.config.context_length
        )
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of
        # layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"""The `{mask_name}` should be specified for {len(self.layers)}
                        layers, but it is for"""
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.info(
                        """`use_cache=True` is incompatible with gradient checkpointing.
                        Setting `use_cache=False`..."""
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
