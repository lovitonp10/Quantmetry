from typing import Optional

import configs
import torch
import torch.nn as nn
from domain.submodules.tft import (
    FeatureEmbedder,
    GatedResidualNetwork,
    TemporalFusionDecoder,
    TemporalFusionEncoder,
    VariableSelectionNetwork,
)
from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler


class TFTModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        model_config: configs.ModelConfig,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        num_past_feat_dynamic_real: int,
        # univariate input
        distr_output: DistributionOutput = StudentTOutput(),
    ) -> None:
        super().__init__()

        self.model_config = model_config

        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real

        self.model_config.lags_sequence = (
            self.model_config.lags_sequence or get_lags_for_frequency(freq_str=freq)
        )
        self.history_length = self.model_config.context_length + max(
            self.model_config.lags_sequence
        )
        self.embedder = FeatureEmbedder(
            cardinalities=self.model_config.cardinality,
            embedding_dims=[self.model_config.variable_dim] * num_feat_static_cat,
        )
        if self.model_config.scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        # projection networks
        self.target_proj = nn.Linear(
            in_features=self.model_config.input_size * len(self.model_config.lags_sequence),
            out_features=self.model_config.variable_dim,
        )

        self.dynamic_proj = nn.Linear(
            in_features=num_feat_dynamic_real, out_features=self.model_config.variable_dim
        )

        if self.num_past_feat_dynamic_real > 0:
            self.past_dynamic_proj = nn.Linear(
                in_features=num_past_feat_dynamic_real, out_features=self.model_config.variable_dim
            )
        else:
            self.past_dynamic_proj = None

        self.static_feat_proj = nn.Linear(
            in_features=num_feat_static_real + self.model_config.input_size,
            out_features=self.model_config.variable_dim,
        )

        # variable selection networks
        self.past_selection = VariableSelectionNetwork(
            d_hidden=self.model_config.variable_dim,
            n_vars=3
            if self.num_past_feat_dynamic_real > 0
            else 2,  # target, time features (and past_feat_dynamic_real)
            dropout=self.model_config.dropout,
            add_static=True,
        )

        self.future_selection = VariableSelectionNetwork(
            d_hidden=self.model_config.variable_dim,
            n_vars=2,  # target and time features
            dropout=self.model_config.dropout,
            add_static=True,
        )

        self.static_selection = VariableSelectionNetwork(
            d_hidden=self.model_config.variable_dim,
            n_vars=self.num_feat_static_cat + 1
            if self.num_feat_static_cat > 1
            else 2,  # 2,   cat, static_feat
            dropout=self.model_config.dropout,
        )

        # Static Gated Residual Networks
        self.selection = GatedResidualNetwork(
            d_hidden=self.model_config.variable_dim,
            dropout=self.model_config.dropout,
        )

        self.enrichment = GatedResidualNetwork(
            d_hidden=self.model_config.variable_dim,
            dropout=self.model_config.dropout,
        )

        self.state_h = GatedResidualNetwork(
            d_hidden=self.model_config.variable_dim,
            d_output=self.model_config.d_models,
            dropout=self.model_config.dropout,
        )

        self.state_c = GatedResidualNetwork(
            d_hidden=self.model_config.variable_dim,
            d_output=self.model_config.d_models,
            dropout=self.model_config.dropout,
        )

        # Encoder and Decoder network
        self.temporal_encoder = TemporalFusionEncoder(
            d_input=self.model_config.variable_dim,
            d_hidden=self.model_config.d_models,
        )
        self.temporal_decoder = TemporalFusionDecoder(
            context_length=self.model_config.context_length,
            prediction_length=self.model_config.prediction_length,
            d_hidden=self.model_config.d_models,
            d_var=self.model_config.variable_dim,
            num_heads=self.model_config.num_heads,
            dropout=self.model_config.dropout,
        )

        # distribution output
        self.param_proj = distr_output.get_args_proj(self.model_config.d_models)

    @property
    def _past_length(self) -> int:
        return self.model_config.context_length + max(self.model_config.lags_sequence)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        shift: int
            shift the lags by this amount back.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.model_config.lags_sequence]

        assert max(indices) + subsequences_length <= sequence_length, (
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
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        # time feature
        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, self._past_length - self.model_config.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.model_config.context_length :, ...]
        )

        # calculate scale
        context = past_target[:, -self.model_config.context_length :]
        observed_context = past_observed_values[:, -self.model_config.context_length :]
        _, scale = self.scaler(context, observed_context)

        # scale the target and create lag features of targets
        target = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )
        subsequences_length = (
            self.model_config.context_length + self.model_config.prediction_length
            if future_target is not None
            else self.model_config.context_length
        )

        lagged_target = self.get_lagged_subsequences(
            sequence=target,
            subsequences_length=subsequences_length,
        )
        lags_shape = lagged_target.shape
        reshaped_lagged_target = lagged_target.reshape(lags_shape[0], lags_shape[1], -1)

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        log_scale = scale.log() if self.model_config.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (feat_static_real, log_scale),
            dim=1,
        )

        # return the network inputs
        return (
            reshaped_lagged_target,  # target
            time_feat,  # dynamic real covariates
            scale,  # scale
            list(embedded_cat),  # static covariates
            static_feat,
        )

    def output_params(self, target, time_feat, embedded_cat, static_feat, past_feat_dynamic_real):
        target_proj = self.target_proj(target)

        past_target_proj = target_proj[:, : self.model_config.context_length, ...]
        future_target_proj = target_proj[:, self.model_config.context_length :, ...]

        time_feat_proj = self.dynamic_proj(time_feat)
        past_time_feat_proj = time_feat_proj[:, : self.model_config.context_length, ...]
        future_time_feat_proj = time_feat_proj[:, self.model_config.context_length :, ...]

        past_selection_list = [past_target_proj, past_time_feat_proj]
        if self.past_dynamic_proj is not None:
            past_dynamic_feat_projs = self.past_dynamic_proj(past_feat_dynamic_real)[
                :, : self.model_config.context_length, ...
            ]
            past_selection_list.append(past_dynamic_feat_projs)

        static_feat_proj = self.static_feat_proj(static_feat)

        static_var, _ = self.static_selection(embedded_cat + [static_feat_proj])
        static_selection = self.selection(static_var).unsqueeze(1)
        static_enrichment = self.enrichment(static_var).unsqueeze(1)

        past_selection, _ = self.past_selection(past_selection_list, static_selection)

        future_selection, _ = self.future_selection(
            [future_target_proj, future_time_feat_proj], static_selection
        )

        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)
        states = [c_h.unsqueeze(0), c_c.unsqueeze(0)]

        enc_out = self.temporal_encoder(past_selection, future_selection, states)

        dec_output = self.temporal_decoder(enc_out, static_enrichment)

        return self.param_proj(dec_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

    # for prediction
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        past_dynamic_real: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        if num_parallel_samples is None:
            num_parallel_samples = self.model_config.num_parallel_samples

        (target, time_feat, scale, embedded_cat, static_feat,) = self.create_network_inputs(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
        )

        past_target_proj = self.target_proj(target)
        past_time_feat_proj = self.dynamic_proj(time_feat)
        future_time_feat_proj = self.dynamic_proj(future_time_feat)
        static_feat_proj = self.static_feat_proj(static_feat)

        static_var, _ = self.static_selection(embedded_cat + [static_feat_proj])
        static_selection = self.selection(static_var).unsqueeze(1)
        static_enrichment = self.enrichment(static_var).unsqueeze(1)

        past_selection_list = [past_target_proj, past_time_feat_proj]
        if self.past_dynamic_proj is not None:
            past_dynamic_feat_projs = self.past_dynamic_proj(past_dynamic_real)[
                :, : self.model_config.context_length, ...
            ]
            past_selection_list.append(past_dynamic_feat_projs)

        past_selection, _ = self.past_selection(past_selection_list, static_selection)

        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)
        states = [c_h.unsqueeze(0), c_c.unsqueeze(0)]

        repeated_scale = scale.repeat_interleave(
            repeats=self.model_config.num_parallel_samples, dim=0
        )
        repeated_time_feat_proj = future_time_feat_proj.repeat_interleave(
            repeats=self.model_config.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.model_config.num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_past_selection = past_selection.repeat_interleave(
            repeats=self.model_config.num_parallel_samples, dim=0
        )
        repeated_static_selection = static_selection.repeat_interleave(
            repeats=self.model_config.num_parallel_samples, dim=0
        )
        repeated_static_enrichment = static_enrichment.repeat_interleave(
            repeats=self.model_config.num_parallel_samples, dim=0
        )
        repeated_states = [
            s.repeat_interleave(repeats=self.model_config.num_parallel_samples, dim=1)
            for s in states
        ]

        # greedy decoding
        future_samples = []
        for k in range(self.model_config.prediction_length):
            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )
            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
            reshaped_lagged_sequence_proj = self.target_proj(reshaped_lagged_sequence)

            next_time_feat_proj = repeated_time_feat_proj[:, : k + 1]
            future_selection, _ = self.future_selection(
                [reshaped_lagged_sequence_proj, next_time_feat_proj],
                repeated_static_selection,
            )
            enc_out = self.temporal_encoder(
                repeated_past_selection, future_selection, repeated_states
            )

            dec_output = self.temporal_decoder(enc_out, repeated_static_enrichment, causal=False)

            params = self.param_proj(dec_output[:, -1:])
            distr = self.output_distribution(params, scale=repeated_scale)

            next_sample = distr.sample()
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.model_config.num_parallel_samples, self.model_config.prediction_length)
            + self.target_shape,
        )
