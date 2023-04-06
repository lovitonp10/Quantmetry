from typing import List, Optional, Tuple
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder as BaseFeatureEmbedder
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler


class FeatureEmbedder(BaseFeatureEmbedder):
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        concat_features = super(FeatureEmbedder, self).forward(features=features)

        if self._num_features > 1:
            features = torch.chunk(concat_features, self._num_features, dim=-1)
        else:
            features = [concat_features]

        return features


class GatedResidualNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        d_input: Optional[int] = None,
        d_output: Optional[int] = None,
        d_static: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        d_input = d_input or d_hidden
        d_static = d_static or 0
        if d_output is None:
            d_output = d_input
            self.add_skip = False
        else:
            if d_output != d_input:
                self.add_skip = True
                self.skip_proj = nn.Linear(in_features=d_input, out_features=d_output)
            else:
                self.add_skip = False

        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_input + d_static, out_features=d_hidden),
            nn.ELU(),
            nn.Linear(in_features=d_hidden, out_features=d_hidden),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_output * 2),
            nn.GLU(),
        )

        self.lnorm = nn.LayerNorm(d_output)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x

        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.mlp(x)
        x = self.lnorm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        n_vars: int,
        dropout: float = 0.0,
        add_static: bool = False,
    ):
        super().__init__()
        self.weight_network = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_input=d_hidden * n_vars,
            d_output=n_vars,
            d_static=d_hidden if add_static else None,
            dropout=dropout,
        )

        self.variable_network = nn.ModuleList(
            [GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout) for _ in range(n_vars)]
        )

    def forward(
        self, variables: List[torch.Tensor], static: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = torch.cat(variables, dim=-1)
        if static is not None:
            static = static.expand_as(variables[0])
        weight = self.weight_network(flatten, static)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)

        var_encodings = [net(var) for var, net in zip(variables, self.variable_network)]
        var_encodings = torch.stack(var_encodings, dim=-1)

        var_encodings = torch.sum(var_encodings * weight, dim=-1)

        return var_encodings, weight


class TemporalFusionEncoder(nn.Module):
    @validated()
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        if d_input != d_hidden:
            self.skip_proj = nn.Linear(in_features=d_input, out_features=d_hidden)
            self.add_skip = True
        else:
            self.add_skip = False

        self.lnorm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        ctx_input: torch.Tensor,
        tgt_input: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
    ):
        ctx_encodings, states = self.encoder_lstm(ctx_input, states)

        if tgt_input is not None:
            tgt_encodings, _ = self.decoder_lstm(tgt_input, states)
            encodings = torch.cat((ctx_encodings, tgt_encodings), dim=1)
            skip = torch.cat((ctx_input, tgt_input), dim=1)
        else:
            encodings = ctx_encodings
            skip = ctx_input

        if self.add_skip:
            skip = self.skip_proj(skip)
        encodings = self.gate(encodings)
        encodings = self.lnorm(skip + encodings)
        return encodings


class TemporalFusionDecoder(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_hidden: int,
        d_var: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.enrich = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_static=d_var,
            dropout=dropout,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.att_net = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.att_lnorm = nn.LayerNorm(d_hidden)

        self.ff_net = nn.Sequential(
            GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.ff_lnorm = nn.LayerNorm(d_hidden)

        self.register_buffer(
            "attn_mask",
            self._generate_subsequent_mask(prediction_length, prediction_length + context_length),
        )

    @staticmethod
    def _generate_subsequent_mask(target_length: int, source_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(source_length, target_length)) == 1).transpose(0, 1)
        mask = (
            mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        expanded_static = static.expand_as(x)
        # static.repeat((1, self.context_length + self.prediction_length, 1))

        skip = x[:, self.context_length :, ...]
        x = self.enrich(x, expanded_static)

        # does not work on GPU :-(
        # mask_pad = torch.ones_like(mask)[:, 0:1, ...]
        # mask_pad = mask_pad.repeat((1, self.prediction_length))
        # key_padding_mask = torch.cat((mask, mask_pad), dim=1).bool()

        query_key_value = x
        attn_output, _ = self.attention(
            query=query_key_value[:, self.context_length :, ...],
            key=query_key_value,
            value=query_key_value,
            # key_padding_mask=key_padding_mask,
            attn_mask=self.attn_mask if causal else None,
        )
        att = self.att_net(attn_output)

        x = x[:, self.context_length :, ...]
        x = self.att_lnorm(x + att)
        x = self.ff_net(x)
        x = self.ff_lnorm(x + skip)

        return x


class TFTModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        # TFT inputs
        num_heads: int,
        embed_dim: int,
        variable_dim: int,
        dropout: float,
        # univariate input
        input_size: int = 1,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        self.input_size = input_size

        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real

        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=[variable_dim] * num_feat_static_cat,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output

        # projection networks
        self.target_proj = nn.Linear(
            in_features=input_size * len(self.lags_seq), out_features=variable_dim
        )

        self.dynamic_proj = nn.Linear(in_features=num_feat_dynamic_real, out_features=variable_dim)

        self.static_feat_proj = nn.Linear(
            in_features=num_feat_static_real + input_size, out_features=variable_dim
        )

        # variable selection networks
        self.past_selection = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=2,  # target and time features
            dropout=dropout,
            add_static=True,
        )

        self.future_selection = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=2,  # target and time features
            dropout=dropout,
            add_static=True,
        )

        self.static_selection = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=2,  # cat, static_feat
            dropout=dropout,
        )

        # Static Gated Residual Networks
        self.selection = GatedResidualNetwork(
            d_hidden=variable_dim,
            dropout=dropout,
        )

        self.enrichment = GatedResidualNetwork(
            d_hidden=variable_dim,
            dropout=dropout,
        )

        self.state_h = GatedResidualNetwork(
            d_hidden=variable_dim,
            d_output=embed_dim,
            dropout=dropout,
        )

        self.state_c = GatedResidualNetwork(
            d_hidden=variable_dim,
            d_output=embed_dim,
            dropout=dropout,
        )

        # Encoder and Decoder network
        self.temporal_encoder = TemporalFusionEncoder(
            d_input=variable_dim,
            d_hidden=embed_dim,
        )
        self.temporal_decoder = TemporalFusionDecoder(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_hidden=embed_dim,
            d_var=variable_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # distribution output
        self.param_proj = distr_output.get_args_proj(embed_dim)

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

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
        indices = [lag - shift for lag in self.lags_seq]

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
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # calculate scale
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        # scale the target and create lag features of targets
        target = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )
        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )

        lagged_target = self.get_lagged_subsequences(
            sequence=target,
            subsequences_length=subsequences_length,
        )
        lags_shape = lagged_target.shape
        reshaped_lagged_target = lagged_target.reshape(lags_shape[0], lags_shape[1], -1)

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (feat_static_real, log_scale),
            dim=1,
        )

        # return the network inputs
        return (
            reshaped_lagged_target,  # target
            time_feat,  # dynamic real covariates
            scale,  # scale
            embedded_cat,  # static covariates
            static_feat,
        )

    def output_params(self, target, time_feat, embedded_cat, static_feat):
        target_proj = self.target_proj(target)

        past_target_proj = target_proj[:, : self.context_length, ...]
        future_target_proj = target_proj[:, self.context_length :, ...]

        time_feat_proj = self.dynamic_proj(time_feat)
        past_time_feat_proj = time_feat_proj[:, : self.context_length, ...]
        future_time_feat_proj = time_feat_proj[:, self.context_length :, ...]

        static_feat_proj = self.static_feat_proj(static_feat)

        static_var, _ = self.static_selection(embedded_cat + [static_feat_proj])
        static_selection = self.selection(static_var).unsqueeze(1)
        static_enrichment = self.enrichment(static_var).unsqueeze(1)

        past_selection, _ = self.past_selection(
            [past_target_proj, past_time_feat_proj], static_selection
        )

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
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

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

        past_selection, _ = self.past_selection(
            [past_target_proj, past_time_feat_proj], static_selection
        )

        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)
        states = [c_h.unsqueeze(0), c_c.unsqueeze(0)]

        repeated_scale = scale.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        repeated_time_feat_proj = future_time_feat_proj.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_past_selection = past_selection.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_selection = static_selection.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_enrichment = static_enrichment.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_states = [
            s.repeat_interleave(repeats=self.num_parallel_samples, dim=1) for s in states
        ]

        # greedy decoding
        future_samples = []
        for k in range(self.prediction_length):
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
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape,
        )


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(
                device
            )

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log1p(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log1p(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class InformerModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        # Informer arguments
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        attn: str = "prob",
        factor: int = 5,
        distil: bool = True,
        # univariate input
        input_size: int = 1,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        self.input_size = input_size

        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        # total feature size
        d_model = self.input_size * len(self.lags_seq) + self._number_of_features

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)

        # Informer enc-decoder
        Attn = ProbAttention if attn == "prob" else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        nhead,
                        mix=False,
                    ),
                    d_model,
                    d_ff=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(num_encoder_layers)
            ],
            [ConvLayer(d_model) for i in range(num_encoder_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Masked Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            mask_flag=True,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        nhead,
                        mix=True,
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        nhead,
                        mix=False,
                    ),
                    d_model,
                    d_ff=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(num_decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

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
        indices = [lag - shift for lag in self.lags_seq]

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

    def _check_shapes(
        self,
        prior_input: torch.Tensor,
        inputs: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> None:
        assert len(prior_input.shape) == len(inputs.shape)
        assert (len(prior_input.shape) == 2 and self.input_size == 1) or prior_input.shape[
            2
        ] == self.input_size
        assert (len(inputs.shape) == 2 and self.input_size == 1) or inputs.shape[
            -1
        ] == self.input_size
        assert (
            features is None or features.shape[2] == self._number_of_features
        ), f"{features.shape[2]}, expected {self._number_of_features}"

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
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # target
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        inputs = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )

        inputs_length = (
            self._past_length + self.prediction_length
            if future_target is not None
            else self._past_length
        )
        assert inputs.shape[1] == inputs_length

        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_scale),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # self._check_shapes(prior_input, inputs, features)

        # sequence = torch.cat((prior_input, inputs), dim=1)
        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs,
            subsequences_length=subsequences_length,
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        return transformer_inputs, scale, static_feat

    def output_params(self, transformer_inputs):
        enc_input = transformer_inputs[:, : self.context_length, ...]
        dec_input = transformer_inputs[:, self.context_length :, ...]

        enc_out, _ = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_out)

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
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:

        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        encoder_inputs, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
        )

        enc_out, _ = self.encoder(encoder_inputs)

        repeated_scale = scale.repeat_interleave(repeats=self.num_parallel_samples, dim=0)

        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
            / repeated_scale
        )

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_feat.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(repeats=self.num_parallel_samples, dim=0)

        repeated_enc_out = enc_out.repeat_interleave(repeats=self.num_parallel_samples, dim=0)

        future_samples = []

        # greedy decoding
        for k in range(self.prediction_length):
            # self._check_shapes(repeated_past_target, next_sample, next_features)
            # sequence = torch.cat((repeated_past_target, next_sample), dim=1)

            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1
            )

            output = self.decoder(decoder_input, repeated_enc_out)

            params = self.param_proj(output[:, -1:])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()

            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape,
        )
