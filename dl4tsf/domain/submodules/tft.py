from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.torch.modules.feature import FeatureEmbedder as BaseFeatureEmbedder


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
