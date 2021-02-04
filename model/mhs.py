from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm, CrossEntropyLoss
from torch.nn.functional import relu, gelu

from module.entity_aware_transformer import EATransformerEncoderLayer, EATransformerEncoder
from module.selection import INFERENCE_CLASS
from utils.bert_utils import bert_mask_wrapper, bert_sequence_output_wrapper
from pytorch_transformers import BertPreTrainedModel, BertModel

import torch.nn.functional as F

ACT2FN = {"gelu": gelu, "relu": relu, }


# class MHS(nn.Module):
#     def __init__(self, labels, selection_layer, ):
#         super().__init__()
#         self.labels = labels
#         self.num_labels = len(labels)
#         self.na_label_id = self.labels.index("NA")
#
#         selection_params = {"input_size": hidden_size, "hidden_size": selection_hidden_size,
#                             "num_labels": self.num_labels}
#         if selection_layer == "SelfAttnMHSLayer":
#             selection_params.pop("hidden_size")


class EATransformerMHS(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, selection_layer,
                 dropout=0.1, attn_dropout=0.1, activation="relu", max_distance=2, selection_layer_params=None,
                 mask_entity=True):
        super().__init__()
        self.labels = labels
        self.num_labels = len(labels)
        self.na_label_id = self.labels.index("NA")
        self.mask_entity = mask_entity

        self.sentence_encoder = sentence_encoder
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.sentence_encoder.out_dim

        self.output_attention = False
        all_selection_params = {"input_size": hidden_size, "num_labels": self.num_labels}
        all_selection_params.update(selection_layer_params)

        if selection_layer == "SelfAttnMHSLayer":
            self.output_attention = True

        attn_encoder_layer = EATransformerEncoderLayer(d_model=hidden_size, n_head=num_attention_heads,
                                                       dim_feedforward=dim_feedforward, dropout=attn_dropout,
                                                       activation=activation, output_attentions=self.output_attention,
                                                       output_values_layer=self.output_attention)
        self.transformer_encoder = EATransformerEncoder(attn_encoder_layer, num_layers,
                                                        hidden_size=hidden_size,
                                                        max_distance=max_distance,
                                                        output_attentions=self.output_attention,
                                                        output_rel_states=self.output_attention)
        self.selection_layer = INFERENCE_CLASS[selection_layer](**all_selection_params)
        # self.init_weights()

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

        # transformer_output = self.transformer_encoder(encoder_output, mask,
        #                                               relative_positions=relative_positions,
        #                                               entity_mask=entity_mask
        #                                               )

        if self.output_attention:
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
        else:
            selection_logits = self.selection_layer(self.transformer_encoder(encoder_output, mask,
                                                                             relative_positions=relative_positions,
                                                                             entity_mask=entity_mask
                                                                             )[0])
        if self.mask_entity:
            mask = features[0] != 0
        else:
            mask = mask == 1
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


class EATransformerMHSSoftmax(EATransformerMHS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def masked_loss(self, selection_logits, selection_gold, mask):
        loss_fct = CrossEntropyLoss()
        active_loss = (mask.unsqueeze(2) * mask.unsqueeze(1)).view(-1) == 1
        selection_logits = selection_logits.permute(0, 1, 3, 2).contiguous()
        active_logits = selection_logits.view(-1, self.num_labels)[active_loss]
        active_labels = selection_gold.view(-1)[active_loss]
        selection_loss = loss_fct(active_logits, active_labels)
        return selection_loss

    def inference(self, selection_logits, mask):
        # selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
        #     .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
        # selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5

        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)) != 1
        probs = selection_logits.permute(0, 1, 3, 2).detach()
        output = torch.argmax(probs, dim=-1)
        output[selection_mask] = self.na_label_id
        idx = (output != self.na_label_id).nonzero()
        batch_size = selection_logits.size(0)
        result = [[] for _ in range(batch_size)]

        for i in range(idx.size(0)):
            batch, s, o = idx[i].tolist()
            p = int(output[batch, s, o])
            triple = (s, p, o)
            if p == self.na_label_id:
                continue
            result[batch].append(triple)
        return result


class DepEATransformerMHS(EATransformerMHS):
    def __init__(self, *args, **kwargs):
        super(DepEATransformerMHS, self).__init__(*args, **kwargs)

    def forward(self, input_ids, features: Tuple, gold_selection_matrix=None, relative_positions=None, entity_mask=None,
                attention_mask=None, mask=None, dependency_graph=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.sentence_encoder(input_ids, features, attention_mask)

        if self.output_attention:
            all_attention_output, all_rel_pos_output = self.transformer_encoder(encoder_output, mask,
                                                                                relative_positions=relative_positions,
                                                                                entity_mask=entity_mask
                                                                                )[-2:]
            attention_probs = all_attention_output[-1][0]
            value_attentions = all_attention_output[-1][-1]
            rel_pos_value = all_rel_pos_output[-1]

            selection_logits = self.selection_layer(attention_probs=attention_probs,
                                                    value_attentions=value_attentions,
                                                    relative_positions=rel_pos_value,
                                                    dependency_graph=dependency_graph)
        else:
            attention_output = self.transformer_encoder(encoder_output, mask, relative_positions=relative_positions,
                                                        entity_mask=entity_mask
                                                        )[0]
            selection_logits = self.selection_layer(attention_output, dependency_graph=dependency_graph)
        if self.mask_entity:
            mask = features[0] != 0
        else:
            mask = mask == 1
        if gold_selection_matrix is not None:
            loss = self.masked_loss(selection_logits, gold_selection_matrix, mask)
            return loss,
        else:
            return self.inference(selection_logits, mask),

# class MHSCell(nn.Module):
#     def __init__(self, input_size, hidden_size, relation_num, activation="gelu"):
#         super(MHSCell, self).__init__()
#         self.selection_u = nn.Linear(input_size, hidden_size)
#         self.selection_v = nn.Linear(input_size, hidden_size)
#         self.selection_uv = nn.Linear(2 * hidden_size, hidden_size)
#         self.relation_embedding = nn.Embedding(num_embeddings=relation_num, embedding_dim=hidden_size)
#         self.activation = ACT2FN[activation]
#
#     def forward(self, input_tensor):
#         B, L, H = input_tensor.size()
#         u = self.activation(self.selection_u(input_tensor)).unsqueeze(1).expand(B, L, L, -1)
#         v = self.activation(self.selection_v(input_tensor)).unsqueeze(2).expand(B, L, L, -1)
#         uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))
#         output = torch.einsum('bijh,rh->birj', uv, self.relation_embedding.weight)
#         return output
#
#
# class MultiHeadSelection(BertPreTrainedModel):
#     def __init__(self, config):
#         super(MultiHeadSelection, self).__init__(config)
#
#         self.num_labels = len(config.labels)
#         self.num_relations = len(config.relations)
#
#         self.relations = config.relations
#         self.na_relation = self.relations.index("NA")
#
#         self.rel_embedding_dim = config.rel_embedding_dim
#         self.label_embedding_dim = config.label_embedding_dim
#         self.label_embedding = nn.Embedding(num_embeddings=self.num_labels, embedding_dim=self.label_embedding_dim)
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.bert = BertModel(config)
#
#         self.feature_mod = config.feature_mod
#
#         if self.feature_mod == "sum":
#             self.mhs_input_size = config.hidden_size
#         else:
#             self.bert2hidden = nn.Linear(config.hidden_size, config.label_embedding_dim)
#             self.mhs_input_size = self.label_embedding_dim * 2
#
#         self.selection_cell = MHSCell(self.mhs_input_size, self.rel_embedding_dim, self.num_relations, config.hidden_act)
#         self.init_weights()
#
#     def forward(self, input_ids, labels, gold_selection_matrix=None, mask=None, attention_mask=None,
#                 token_type_ids=None, position_ids=None, head_mask=None):
#         if mask is None and attention_mask is not None:
#             mask = bert_mask_wrapper(attention_mask)
#             mask = mask == 1
#
#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask)
#
#         sequence_output = bert_sequence_output_wrapper(outputs[0])
#
#         if self.feature_mod == "sum":
#             sequence_output = sequence_output + self.label_embedding(labels)
#         else:
#             sequence_output = self.bert2hidden(sequence_output)
#             sequence_output = torch.cat((sequence_output, self.label_embedding(labels)), -1)
#
#         sequence_output = self.dropout(sequence_output)
#         selection_logits = self.selection_cell(sequence_output)
#
#         if gold_selection_matrix is not None:
#             loss = self.masked_loss(selection_logits, gold_selection_matrix, mask)
#             return loss,
#         else:
#             return self.inference(selection_logits, mask),
#
#     def masked_loss(self, selection_logits, selection_gold, mask):
#         selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
#             .expand(-1, -1, self.num_relations, -1)  # batch x seq x rel x seq
#         selection_loss = F.binary_cross_entropy_with_logits(selection_logits, selection_gold, reduction='none')
#         selection_loss = selection_loss.masked_select(selection_mask).sum()
#         selection_loss /= mask.sum()
#         return selection_loss
#
#     def inference(self, selection_logits, mask):
#         selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
#             .expand(-1, -1, self.num_relations, -1)  # batch x seq x rel x seq
#         selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5
#
#         batch_size = len(selection_tags)
#         result = [[] for _ in range(batch_size)]
#         idx = torch.nonzero(selection_tags.cpu())
#         for i in range(idx.size(0)):
#             batch, s, p, o = idx[i].tolist()
#             triple = (s, p, o)
#             if p == self.na_relation:
#                 continue
#             result[batch].append(triple)
#         return result
