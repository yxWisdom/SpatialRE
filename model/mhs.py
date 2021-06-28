from collections import defaultdict
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm, CrossEntropyLoss, TransformerEncoderLayer, TransformerEncoder
from torch.nn.functional import relu, gelu

from model.sequence_labeling import LSTMCRFSeqLabeling
from module.entity_aware_transformer import EATransformerEncoderLayer, EATransformerEncoder
from module.lstm import LSTM
from module.selection import INFERENCE_CLASS
from utils.bert_utils import bert_mask_wrapper, entity_relative_distance_matrix

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



class BERTMHS(nn.Module):
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
        self.bi_lstm = LSTM(input_size=hidden_size,
                            hidden_size=hidden_size//2,
                            num_layers=2, bidirectional=True)

        all_selection_params = {"input_size": hidden_size, "num_labels": self.num_labels}
        all_selection_params.update(selection_layer_params)
        self.selection_layer = INFERENCE_CLASS[selection_layer](**all_selection_params)

    def forward(self, input_ids, features: Tuple, gold_selection_matrix=None, relative_positions=None, entity_mask=None,
                attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.sentence_encoder(input_ids, features, attention_mask)
        selection_logits = self.selection_layer(encoder_output)

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

class LSTMMHS(nn.Module):
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
        self.bi_lstm = LSTM(input_size=hidden_size,
                            hidden_size=hidden_size//2,
                            num_layers=2, bidirectional=True)

        all_selection_params = {"input_size": hidden_size, "num_labels": self.num_labels}
        all_selection_params.update(selection_layer_params)
        self.selection_layer = INFERENCE_CLASS[selection_layer](**all_selection_params)

    def forward(self, input_ids, features: Tuple, gold_selection_matrix=None, relative_positions=None, entity_mask=None,
                attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.sentence_encoder(input_ids, features, attention_mask)

        # transformer_output = self.transformer_encoder(encoder_output, mask,
        #                                               relative_positions=relative_positions,
        #                                               entity_mask=entity_mask
        #                                               )

        seq_lengths = torch.sum(mask, -1)
        encoder_output = self.bi_lstm(encoder_output, seq_len=seq_lengths)[0]
        selection_logits = self.selection_layer(encoder_output)

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

class TransformerMHS(nn.Module):
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

        attn_encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads)
        self.transformer_encoder = TransformerEncoder(attn_encoder_layer, num_layers)
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
        attn_mask = mask == 1
        encoder_output = self.transformer_encoder(encoder_output.transpose(0, 1), src_key_padding_mask=attn_mask)\
            .transpose(0, 1)
        selection_logits = self.selection_layer(encoder_output)

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


class Config3BModel1(EATransformerMHS):
    def __init__(self, *args, label_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.element_labels = label_dict['element_label']
        self.qs_rel_types = label_dict['qs_rel_type']
        self.o_rel_types = label_dict['o_rel_type']

    def inference(self, selection_logits, mask):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
            .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5

        batch_size = len(selection_tags)
        # (link_triples, rel_type_triples)
        result = [list() for _ in range(batch_size)]
        idx = torch.nonzero(selection_tags.cpu())

        qs_dicts = [defaultdict(lambda: ('', 0)) for _ in range(batch_size)]
        o_dicts = [defaultdict(lambda: ('', 0)) for _ in range(batch_size)]
        for i in range(idx.size(0)):
            batch, s, p, o = idx[i].tolist()
            if p == self.na_label_id:
                continue
            prob = selection_logits[batch, s, p, o].item()
            key = (s, o)
            if self.labels[p] in self.qs_rel_types:
                if qs_dicts[batch][key][1] < prob:
                    qs_dicts[batch][key] = (p, prob)
            elif self.labels[p] in self.o_rel_types:
                if o_dicts[batch][key][1] < prob:
                    o_dicts[batch][key] = (p, prob)
            else:
                result[batch].append((s, p, o))
        for batch, dict_ in enumerate(qs_dicts + o_dicts):
            batch = batch % batch_size
            for (s, o), (p, _) in dict_.items():
                result[batch].append((s, p, o))
        return result

    # def inference(self, selection_logits, mask):
    #     selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
    #         .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
    #     selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5
    #
    #     batch_size = len(selection_tags)
    #     # (link_triples, rel_type_triples)
    #     result = [([], []) for _ in range(batch_size)]
    #     idx = torch.nonzero(selection_tags.cpu())
    #     for i in range(idx.size(0)):
    #         batch, s, p, o = idx[i].tolist()
    #         if p == self.na_label_id:
    #             continue
    #         if s == o:
    #             prob = selection_logits[batch, s, p, o].item()
    #             result[batch][1].append((s, p, prob))
    #         else:
    #             result[batch][0].append((s, p, o))
    #     return result


# class Config3BModel2(EA)


class Config3BModel2Softmax(EATransformerMHS):
    def __init__(self, *args, label_dict=None, **kwargs):
        super().__init__(*args, **kwargs)

        # self.element_labels = element_labels
        # self.qs_rel_types = qs_rel_types
        # self.o_rel_types = o_rel_types

        self.element_labels = label_dict['element_label']
        self.qs_rel_types = label_dict['qs_rel_type']
        self.o_rel_types = label_dict['o_rel_type']
        self.no_trigger_q_labels = label_dict['no_trigger_q_rel_type']
        self.no_trigger_o_labels = label_dict['no_trigger_o_rel_type']
        self.num_qs_rel = len(self.qs_rel_types)
        self.num_o_rel = len(self.o_rel_types)

        hidden_size = self.sentence_encoder.out_dim
        self.qs_linear = nn.Linear(hidden_size, self.num_qs_rel)
        self.o_linear = nn.Linear(hidden_size, self.num_o_rel)

        self.spatial_signals = []
        for idx, label in enumerate(self.element_labels):
            if label[2:].startswith("SPATIAL_SIGNAL"):
                self.spatial_signals.append(idx)

        all_q_selection_params = {"input_size": hidden_size, "num_labels": len(self.no_trigger_q_labels)}
        all_o_selection_params = {"input_size": hidden_size, "num_labels": len(self.no_trigger_o_labels)}
        all_q_selection_params.update(kwargs['selection_layer_params'])
        all_o_selection_params.update(kwargs['selection_layer_params'])
        self.q_selection_layer = INFERENCE_CLASS[kwargs['selection_layer']](**all_q_selection_params)
        self.o_selection_layer = INFERENCE_CLASS[kwargs['selection_layer']](**all_o_selection_params)

        self.no_trigger_q_id = self.labels.index("QSLINK")
        self.no_trigger_o_id = self.labels.index("OLINK")

    def forward(self, input_ids, features: Tuple, gold_selection_matrix=None, relative_positions=None, entity_mask=None,
                attention_mask=None, mask=None, gold_qs_rel_types=None, gold_o_rel_types=None,
                gold_no_trigger_q=None, gold_no_trigger_o=None):
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
                                                    relative_positions=rel_pos_value)
            q_logits = self.q_selection_layer(attention_probs=attention_probs,
                                              value_attentions=value_attentions,
                                              relative_positions=rel_pos_value)
            o_logits = self.o_selection_layer(attention_probs=attention_probs,
                                              value_attentions=value_attentions,
                                              relative_positions=rel_pos_value)
        else:
            input_embed = self.transformer_encoder(encoder_output, mask, relative_positions=relative_positions,
                                                   entity_mask=entity_mask)[0]
            selection_logits = self.selection_layer(input_embed)
            q_logits = self.q_selection_layer(input_embed)
            o_logits = self.o_selection_layer(input_embed)

        qs_rel_type_logits = self.qs_linear(encoder_output)
        o_rel_type_logits = self.o_linear(encoder_output)

        q_logits = q_logits.permute(0, 1, 3, 2)  # .contiguous()
        o_logits = o_logits.permute(0, 1, 3, 2)  # .contiguous()

        if self.mask_entity:
            mask = features[0] != 0
            # for item in self.spatial_signals[1:]:
            #     rel_type_mask = rel_type_mask | features[0] == item
            # rel_type_mask = features[0] != self.element_labels.index("B-SPATIAL_SIGNAL")
        else:
            mask = mask == 1
            # rel_type_mask = mask
        # TODO: 此处需注意
        rel_type_mask = (features[0] >= self.spatial_signals[0]) & (features[0] <= self.spatial_signals[-1])

        if gold_selection_matrix is not None:
            link_loss = self.bce_loss(selection_logits, gold_selection_matrix, mask)
            qs_loss = self.softmax_loss(qs_rel_type_logits, gold_qs_rel_types, rel_type_mask)
            o_loss = self.softmax_loss(o_rel_type_logits, gold_o_rel_types, rel_type_mask)

            q_mask = gold_selection_matrix[:, :, self.no_trigger_q_id, :] != 0
            o_mask = gold_selection_matrix[:, :, self.no_trigger_o_id, :] != 0

            # if q_mask.sum().item() != 0 and o_mask.sum().item() != 0:
            #     print()

            no_trigger_qs_loss = self.softmax_loss(q_logits, gold_no_trigger_q, q_mask)
            no_trigger_o_loss = self.softmax_loss(o_logits, gold_no_trigger_o, o_mask)

            return link_loss + qs_loss + o_loss + no_trigger_o_loss + no_trigger_qs_loss,
        else:
            pred_results = []
            selection_logits = self.selection_output(selection_logits, mask)

            link_pred = self.link_inference(selection_logits, mask)
            qs_rel_type_pred = self.rel_type_inference(qs_rel_type_logits, rel_type_mask, "QSLINK")
            o_rel_type_pred = self.rel_type_inference(o_rel_type_logits, rel_type_mask, "OLINK")

            q_mask = selection_logits[:, :, self.no_trigger_q_id, :] > 0.5
            o_mask = selection_logits[:, :, self.no_trigger_o_id, :] > 0.5

            # if q_mask.sum().item() != 0 or o_mask.sum().item() != 0:
            #     print()

            no_trigger_q_pred = self.no_trigger_inference(q_logits, q_mask, "QSLINK")
            no_trigger_o_pred = self.no_trigger_inference(o_logits, o_mask, "OLINK")

            for a, b, c, d, e in zip(link_pred, qs_rel_type_pred, o_rel_type_pred, no_trigger_q_pred,
                                     no_trigger_o_pred):
                pred_results.append(a + b + c + d + e)
            return pred_results,

    def selection_output(self, selection_logits, mask):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
            .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
        return torch.sigmoid(selection_logits) * selection_mask.float()

    def bce_loss(self, selection_logits, selection_gold, mask):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
            .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits, selection_gold, reduction='none')
        loss = selection_loss.masked_select(selection_mask).sum()
        # TODO: mask or selection_mask ?
        cnt = mask.sum().item()
        if cnt != 0:
            loss /= cnt

        # if cnt == 0:
        #     import sys
        #     sys.stderr.write("cnt为0")
        #     exit(2)

        return loss

    @staticmethod
    def softmax_loss(inputs, labels, mask):
        num_labels = inputs.size(-1)
        # loss_fct = CrossEntropyLoss()
        # active_loss = mask.view(-1) == 1
        # active_logits = inputs.view(-1, num_labels)[active_loss]
        # active_labels = labels.view(-1)[active_loss]
        # loss = loss_fct(active_logits, active_labels)

        logits = inputs.view(-1, num_labels)
        labels = labels.view(-1)
        mask = mask.view(-1)
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_mat = loss_fct(logits, labels)
        loss = loss_mat.masked_select(mask).sum()
        cnt = mask.sum().item()
        if cnt != 0:
            loss /= cnt
        if torch.isnan(loss):
            return 0.0
        return loss

    def rel_type_inference(self, inputs, mask, link_type):
        rel_type_labels = self.qs_rel_types if link_type == "QSLINK" else self.o_rel_types
        pad_label_id = rel_type_labels.index("O")
        inputs = inputs.detach()
        output = torch.argmax(inputs, -1)
        mask = ~mask
        output[mask] = pad_label_id
        indices = (output != pad_label_id).nonzero()
        batch_size = inputs.size(0)
        result = [[] for _ in range(batch_size)]
        # result[1].append([1, 2])
        for i in range(indices.size(0)):
            batch, idx = indices[i].tolist()
            label_id = int(output[batch, idx])
            if label_id == pad_label_id:
                continue
            result[batch].append((idx, rel_type_labels[label_id]))
        return result

    def no_trigger_inference(self, inputs, mask, link_type):
        if link_type == 'QSLINK':
            labels = self.no_trigger_q_labels
        else:
            labels = self.no_trigger_o_labels

        batch_size = inputs.size(0)
        result = [[] for _ in range(batch_size)]
        output = torch.argmax(inputs, dim=-1)
        for b, s, o in mask.nonzero():
            rel = labels[output[b, s, o]]
            result[b].append((s, rel, o))
        return result

    def link_inference(self, selection_logits, mask):
        # selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
        #     .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
        # selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5

        selection_tags = selection_logits > 0.5
        batch_size = len(selection_tags)
        result = [[] for _ in range(batch_size)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            batch, s, p, o = idx[i].tolist()
            if p == self.na_label_id:
                continue
            rel = self.labels[p]
            if rel == "QSLINK" or rel == "OLINK":
                continue
            result[batch].append((s, rel, o))
        return result


class Config1Model1(Config3BModel1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.sentence_encoder.out_dim
        self.ner_layer = LSTMCRFSeqLabeling(hidden_size, self.element_labels)
        self.na_element_id = self.element_labels.index("O")
        self.max_distance = kwargs['max_distance']

    def get_relative_distance(self, batch_pred_elements, batch_token_to_pre_idx):
        batch_relative_positions = []
        batch_entity_mask = []
        _, seq_len = batch_token_to_pre_idx.shape
        for pred_elements, token_to_pre_idx in zip(batch_pred_elements, batch_token_to_pre_idx):
            # pred_elements = [self.element_labels[i] for i in pred_element_ids]
            token_to_pre_idx = token_to_pre_idx.cpu().numpy().tolist()
            relative_positions, entity_mask = entity_relative_distance_matrix(pred_elements,
                                                                              token_to_pre_idx,
                                                                              self.max_distance,
                                                                              seq_len)
            batch_relative_positions.append(relative_positions)
            batch_entity_mask.append(entity_mask)

        relative_positions_tensor = torch.tensor(batch_relative_positions, dtype=torch.long)
        entity_mask_tensor = torch.tensor(batch_entity_mask, dtype=torch.long)

        device = batch_token_to_pre_idx.device
        relative_positions_tensor = relative_positions_tensor.to(device)
        entity_mask_tensor = entity_mask_tensor.to(device)
        return relative_positions_tensor, entity_mask_tensor

    def forward(self, input_ids, gold_element_labels=None, mask=None, attention_mask=None, token_to_pre_idx=None,
                relative_positions=None, entity_mask=None, **kwargs):

        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        bert_output = self.sentence_encoder(input_ids, only_bert=True)
        if gold_element_labels is not None:
            element_loss = self.ner_layer(bert_output, gold_element_labels, mask=mask)
            link_loss = super().forward(input_ids, mask=mask, attention_mask=attention_mask,
                                        relative_positions=relative_positions, entity_mask=entity_mask, **kwargs)

            return element_loss + link_loss[0],
        else:
            batch_pred_element_ids = self.ner_layer(bert_output, mask=mask)
            batch_pred_element_ids[mask != 1] = self.na_element_id
            batch_pred_element_ids = batch_pred_element_ids.detach()
            # batch_pred_element_ids = batch_pred_element_ids.cpu().numpy().tolist()
            batch_pred_elements = [[self.element_labels[i] for i in batch] for batch in batch_pred_element_ids]
            # entity_mask = (batch_pred_elements != self.na_element_id) & (mask == 1)

            relative_positions_, entity_mask_ = self.get_relative_distance(batch_pred_elements, token_to_pre_idx)

            kwargs["features"] = (batch_pred_element_ids,) + kwargs["features"][1:]
            link_result = super().forward(input_ids, mask=mask, attention_mask=attention_mask,
                                          relative_positions=relative_positions_, entity_mask=entity_mask_, **kwargs)

            return [{"elements": elements, "links": links} for elements, links in
                    zip(batch_pred_elements, link_result[0])],


class Config1Model2(Config3BModel2Softmax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.sentence_encoder.out_dim
        self.ner_layer = LSTMCRFSeqLabeling(hidden_size, self.element_labels)
        self.na_element_id = self.element_labels.index("O")
        self.max_distance = kwargs['max_distance']

    def get_relative_distance(self, batch_pred_elements, batch_token_to_pre_idx):
        batch_relative_positions = []
        batch_entity_mask = []
        _, seq_len = batch_token_to_pre_idx.shape
        for pred_elements, token_to_pre_idx in zip(batch_pred_elements, batch_token_to_pre_idx):
            # pred_elements = [self.element_labels[i] for i in pred_element_ids]
            token_to_pre_idx = token_to_pre_idx.cpu().numpy().tolist()
            relative_positions, entity_mask = entity_relative_distance_matrix(pred_elements,
                                                                              token_to_pre_idx,
                                                                              self.max_distance,
                                                                              seq_len)
            batch_relative_positions.append(relative_positions)
            batch_entity_mask.append(entity_mask)

        relative_positions_tensor = torch.tensor(batch_relative_positions, dtype=torch.long)
        entity_mask_tensor = torch.tensor(batch_entity_mask, dtype=torch.long)

        device = batch_token_to_pre_idx.device
        relative_positions_tensor = relative_positions_tensor.to(device)
        entity_mask_tensor = entity_mask_tensor.to(device)
        return relative_positions_tensor, entity_mask_tensor

    def forward(self, input_ids, gold_element_labels=None, mask=None, attention_mask=None, token_to_pre_idx=None,
                relative_positions=None, entity_mask=None, **kwargs):

        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        bert_output = self.sentence_encoder(input_ids, only_bert=True)
        if gold_element_labels is not None:
            element_loss = self.ner_layer(bert_output, gold_element_labels, mask=mask)
            link_loss = super().forward(input_ids, mask=mask, attention_mask=attention_mask,
                                        relative_positions=relative_positions, entity_mask=entity_mask, **kwargs)
            return element_loss + link_loss[0],
        else:
            batch_pred_element_ids = self.ner_layer(bert_output, mask=mask)
            batch_pred_element_ids[mask != 1] = self.na_element_id
            batch_pred_element_ids = batch_pred_element_ids.detach()
            # batch_pred_element_ids = batch_pred_element_ids.cpu().numpy().tolist()
            batch_pred_elements = [[self.element_labels[i] for i in batch] for batch in batch_pred_element_ids]
            # entity_mask = (batch_pred_elements != self.na_element_id) & (mask == 1)

            relative_positions_, entity_mask_ = self.get_relative_distance(batch_pred_elements, token_to_pre_idx)

            kwargs["features"] = (batch_pred_element_ids,) + kwargs["features"][1:]
            link_result = super().forward(input_ids, mask=mask, attention_mask=attention_mask,
                                          relative_positions=relative_positions_, entity_mask=entity_mask_, **kwargs)

            return [{"elements": elements, "links": links} for elements, links in
                    zip(batch_pred_elements, link_result[0])],

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
