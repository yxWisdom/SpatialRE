import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn.functional import gelu, relu

from module.entity_aware_transformer import EATransformerEncoderLayer, EATransformerEncoder
from utils.bert_utils import bert_mask_wrapper, bert_sequence_output_wrapper
from pytorch_transformers import BertPreTrainedModel, BertModel

import torch.nn.functional as F

from utils.config import Config

ACT2FN = {"gelu": gelu, "relu": relu, }


class MHSEmbedding(nn.Module):
    def __init__(self, config):
        super(MHSEmbedding, self).__init__()
        self.label_embedding = nn.Embedding(len(config.labels), config.label_embedding_dim)
        self.feature_mode = config.feature_mode

        if self.feature_mode == "concat":
            self.embedding_output_dim = config.embedding_in_dim + config.label_embedding_dim
        else:
            assert config.embedding_in_dim == config.label_embedding_dim
            self.embedding_output_dim = config.embedding_in_dim
        self.LayerNorm = LayerNorm(self.embedding_output_dim, eps=config.layer_norm_eps)

    def forward(self, input_tensor, label_ids):
        label_embedding = self.label_embedding(label_ids)
        if self.feature_mode == "concat":
            embedding = torch.cat((input_tensor, label_embedding), -1)
        else:
            embedding = input_tensor + label_embedding
        embedding = self.LayerNorm(embedding)
        return embedding


class MHSEmbeddingWithPos(nn.Module):
    def __init__(self, config):
        super(MHSEmbeddingWithPos, self).__init__()

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.pos_embedding_dim)
        self.label_embedding = nn.Embedding(len(config.labels), config.label_embedding_dim)
        self.feature_mode = config.feature_mode

        if self.feature_mode == "concat":
            self.embedding_output_dim = config.embedding_in_dim + config.label_embedding_dim + config.pos_embedding_dim
        else:
            assert config.embedding_in_dim == config.label_embedding_dim == config.pos_embedding_dim
            self.embedding_output_dim = config.embedding_in_dim
        self.LayerNorm = LayerNorm(self.embedding_output_dim, eps=config.layer_norm_eps)

    def forward(self, input_tensor, label_ids):
        seq_length = input_tensor.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device)
        position_ids = position_ids.unsqueeze(0).expand_as(label_ids)
        position_embeddings = self.position_embeddings(position_ids)

        label_embedding = self.label_embedding(label_ids)

        if self.feature_mode == "concat":
            embedding = torch.cat((input_tensor, position_embeddings, label_embedding), -1)
        else:
            embedding = input_tensor + position_embeddings + label_embedding
        embedding = self.LayerNorm(embedding)
        return embedding


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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

        outputs = (context_layer, attention_probs, mixed_value_layer) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
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


class TransformerMHSCell(nn.Module):
    def __init__(self, config):
        super(TransformerMHSCell, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_relations = len(config.relations)
        self.num_hidden_layers = config.num_hidden_layers

        self.dense = nn.Linear(self.all_head_size, self.num_relations)
        self.encoder = TransformerEncoder(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.num_hidden_layers

        output = self.encoder(input_tensor, attention_mask=extended_attention_mask,
                              head_mask=head_mask)
        # last_hidden_state = output[0]
        all_attentions = output[-1]

        last_hidden_state = all_attentions[-1][-1]
        last_hidden_state = self.transpose_for_scores(last_hidden_state)
        B, H, S, E = last_hidden_state.size()
        last_hidden_state = last_hidden_state.unsqueeze(2).expand(B, H, S, S, E)

        attention_probs = all_attentions[-1][0]
        attention_probs = attention_probs.unsqueeze(-1)

        mhs_output = torch.mul(last_hidden_state, attention_probs)

        new_mhs_output_shape = (B, S, S, E * H)
        mhs_output = mhs_output.permute(0, 3, 2, 1, 4).contiguous()
        mhs_output = mhs_output.view(*new_mhs_output_shape)
        mhs_output = self.dense(mhs_output)
        mhs_output = mhs_output.permute(0, 1, 3, 2)
        return mhs_output


class TransformerMHS(BertPreTrainedModel):
    def __init__(self, bert_config):
        super().__init__(bert_config)
        mhs_config = Config(**bert_config.mhs_params)
        # mhs_config = bert_config.mhs_params
        self.num_relations = len(mhs_config.relations)

        self.relations = mhs_config.relations
        self.na_relation = self.relations.index("NA")

        self.dropout = nn.Dropout(mhs_config.hidden_dropout_prob)
        self.bert = BertModel(bert_config)

        # self.feature_mode = mhs_config.feature_mode

        self.embedding_layer = MHSEmbeddingWithPos(mhs_config) if mhs_config.use_pos else MHSEmbedding(mhs_config)

        if bert_config.hidden_size != mhs_config.embedding_in_dim:
            self.bert2hidden = nn.Linear(bert_config.hidden_size, mhs_config.embedding_in_dim)

        self.hidden_size = self.embedding_layer.embedding_output_dim

        self.selection_cell = TransformerMHSCell(mhs_config)

        self.init_weights()

    def forward(self, input_ids, labels, gold_selection_matrix=None, mask=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)
            mask = mask == 1

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])

        if hasattr(self, "bert2hidden"):
            sequence_output = self.bert2hidden(sequence_output)

        sequence_output = self.embedding_layer(sequence_output, labels)
        sequence_output = self.dropout(sequence_output)

        selection_logits = self.selection_cell(sequence_output, attention_mask=attention_mask, head_mask=head_mask)

        mask = labels != 0
        if gold_selection_matrix is not None:
            loss = self.masked_loss(selection_logits, gold_selection_matrix, mask)
            return loss,
        else:
            return self.inference(selection_logits, mask),

    def masked_loss(self, selection_logits, selection_gold, mask):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
            .expand(-1, -1, self.num_relations, -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits, selection_gold, reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    def inference(self, selection_logits, mask):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
            .expand(-1, -1, self.num_relations, -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5

        batch_size = len(selection_tags)
        result = [[] for _ in range(batch_size)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            batch, s, p, o = idx[i].tolist()
            triple = (s, p, o)
            if p == self.na_relation:
                continue
            result[batch].append(triple)
        return result


class TransformerMHSLayer(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(TransformerMHSLayer, self).__init__()
        self.dense = nn.Linear(hidden_size, num_labels)

    def forward(self, attention_probs, value_attentions, relative_positions=None):
        B, H, S, E = value_attentions.size()
        value_attentions = value_attentions.unsqueeze(2).expand(B, H, S, S, E)

        # if relative_positions is not None:
        #     value_attentions += relative_positions

        attention_probs = attention_probs.unsqueeze(-1)
        output = torch.mul(value_attentions, attention_probs)

        if relative_positions is not None:
            relative_positions = relative_positions.view(B, S, S, H, E).permute(0, 3, 1, 2, 4)
            output = output + torch.mul(relative_positions, attention_probs)

        output = output.permute(0, 3, 2, 1, 4).contiguous().view(B, S, S, E * H)

        output = self.dense(output)
        output = output.permute(0, 1, 3, 2)

        return output


class EATransformerMHS(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, dropout=0.1,
                 attn_dropout=0.1, activation="relu", max_distance=2):
        super().__init__()
        self.labels = labels
        self.num_labels = len(labels)
        self.na_label_id = self.labels.index("NA")

        self.sentence_encoder = sentence_encoder
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.sentence_encoder.out_dim

        attn_encoder_layer = EATransformerEncoderLayer(d_model=hidden_size, n_head=num_attention_heads,
                                                       dim_feedforward=dim_feedforward, dropout=attn_dropout,
                                                       activation=activation, output_attentions=True,
                                                       output_values_layer=True)
        self.transformer_encoder = EATransformerEncoder(attn_encoder_layer, num_layers,
                                                        hidden_size=hidden_size,
                                                        max_distance=max_distance,
                                                        output_attentions=True,
                                                        output_rel_states=True)
        self.selection_layer = TransformerMHSLayer(hidden_size, self.num_labels)

    def init_weights(self):
        def _init_weights(module):
            """ Initialize the weights """
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(_init_weights)

    def forward(self, input_ids, features: Tuple, gold_selection_matrix=None, relative_positions=None, entity_mask=None,
                attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.sentence_encoder(input_ids, features, attention_mask)

        all_attention_output, all_rel_pos_output = self.transformer_encoder(encoder_output, mask,
                                                                            relative_positions=relative_positions,
                                                                            entity_mask=entity_mask
                                                                            )[-2:]
        attention_probs = all_attention_output[-1][0]
        value_attentions = all_attention_output[-1][-1]
        rel_pos_value = all_rel_pos_output[-1]

        selection_logits = self.selection_layer(attention_probs=attention_probs,
                                                value_attentions=value_attentions,
                                                relative_positions=rel_pos_value)

        mask = features[0] != 0
        if gold_selection_matrix is not None:
            loss = self.masked_loss(selection_logits, gold_selection_matrix, mask)
            return loss,
        else:
            return self.inference(selection_logits, mask),

    def masked_loss(self, selection_logits, selection_gold, mask):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
            .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits, selection_gold, reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        if mask.sum().item() != 0:
            selection_loss /= mask.sum()
        return selection_loss

    def inference(self, selection_logits, mask):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
            .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5

        batch_size = len(selection_tags)
        result = [[] for _ in range(batch_size)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            batch, s, p, o = idx[i].tolist()
            triple = (s, p, o)
            if p == self.na_label_id:
                continue
            result[batch].append(triple)
        return result
