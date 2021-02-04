import copy

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn.functional import gelu, relu

import torch.nn.functional as F

import math


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads, attention_dropout=0.1, output_attentions=True,
                 output_values_layer=False):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_attention_heads != 0:  # and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (d_model, num_attention_heads)
            )
        self.output_attentions = output_attentions
        self.output_values_layer = output_values_layer

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(d_model / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).type_as(attention_scores)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
        if self.output_attentions:
            outputs += (attention_probs,)
        if self.output_values_layer:
            outputs += (mixed_value_layer,)

        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1, activation="relu",
                 output_attentions=False, output_values_layer=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, attention_dropout=dropout,
                                                 output_attentions=output_attentions,
                                                 output_values_layer=output_values_layer)
        self.attn_dense = nn.Linear(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.feedforward_dense1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.feedforward_dense2 = nn.Linear(dim_feedforward, d_model)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.self_attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        # self-attention layer
        attention_output = self.attn_dense(attention_output)
        attention_output = self.attn_dropout(attention_output)
        attention_output = self.attn_norm(hidden_states + attention_output)

        # feedforward layer
        output = self.feedforward_dense1(attention_output)
        output = self.activation(output)

        # final output
        output = self.feedforward_dense2(output)
        output = self.feedforward_dropout(output)
        output = self.feedforward_norm(attention_output + output)

        outputs = (output,) + attention_outputs[1:]

        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, output_hidden_states=False,
                 output_attentions=False):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        if head_mask is None:
            head_mask = [None] * len(self.layers)

        for i, layer_module in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1:],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
