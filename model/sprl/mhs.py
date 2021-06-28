from collections import defaultdict
from typing import Tuple

import torch
import torch.nn.functional as F
from allennlp.modules import FeedForward
from torch import nn
from torch.nn import LayerNorm, CrossEntropyLoss
from torch.nn.functional import relu, gelu

from model.mhs import EATransformerMHS
from model.sequence_labeling import LSTMCRFSeqLabeling, CRFSeqLabeling
from module.entity_aware_transformer import EATransformerEncoderLayer, EATransformerEncoder
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

# class SpRLModel_A(EATransformerMHS):
#     def __init__(self, *args, label_dict=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.element_labels = label_dict['element_label']
#         self.rel_labels_dict = label_dict['rel_type_label']
#
#     def inference(self, selection_logits, mask):
#         selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
#             .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
#         selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5
#
#         batch_size = len(selection_tags)
#         # (link_triples, rel_type_triples)
#         result = [list() for _ in range(batch_size)]
#         idx = torch.nonzero(selection_tags.cpu())
#
#         rel_type_result = {
#             k: [defaultdict(lambda: ('', 0)) for _ in range(batch_size)] for k in self.rel_labels_dict
#         }
#
#         qs_dicts = [defaultdict(lambda: ('', 0)) for _ in range(batch_size)]
#         o_dicts = [defaultdict(lambda: ('', 0)) for _ in range(batch_size)]
#         for i in range(idx.size(0)):
#             batch, s, p, o = idx[i].tolist()
#             if p == self.na_label_id:
#                 continue
#             prob = selection_logits[batch, s, p, o].item()
#             key = (s, o)
#
#             for k, rel_types in self.rel_labels_dict.item():
#                 if self.labels[p] in rel_types:
#                     rel_type_result[k][batch][key][1]
#
#             if self.labels[p] in self.qs_rel_types:
#                 if qs_dicts[batch][key][1] < prob:
#                     qs_dicts[batch][key] = (p, prob)
#             elif self.labels[p] in self.o_rel_types:
#                 if o_dicts[batch][key][1] < prob:
#                     o_dicts[batch][key] = (p, prob)
#             else:
#                 result[batch].append((s, p, o))
#         for batch, dict_ in enumerate(qs_dicts + o_dicts):
#             batch = batch % batch_size
#             for (s, o), (p, _) in dict_.items():
#                 result[batch].append((s, p, o))
#         return result


class SpRLModel_B(EATransformerMHS):
    def __init__(self, *args, label_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.element_labels = label_dict['element_label']
        self.rel_labels_dict = label_dict['rel_type_label']
        rel_types = list(sorted(self.rel_labels_dict.keys()))
        self.rel2idx = {rel: idx for idx, rel in enumerate(rel_types)}

        self.ss_b_id = self.element_labels.index("B-SPATIAL_SIGNAL")
        self.ss_i_id = self.element_labels.index("I-SPATIAL_SIGNAL")
        hidden_size = self.sentence_encoder.out_dim

        self.rel_classifiers = nn.ModuleList([
            nn.Sequential(
                FeedForward(input_dim=hidden_size * 2,
                            num_layers=2,
                            hidden_dims=150,
                            activations=F.relu,
                            dropout=0.2),
                nn.Linear(150, len(self.rel_labels_dict[rel]))
            ) for rel in rel_types
        ])

    # def bce_loss(self, selection_logits, selection_gold, mask):
    #     selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2) \
    #         .expand(-1, -1, self.num_labels, -1)  # batch x seq x rel x seq
    #     selection_loss = F.binary_cross_entropy_with_logits(selection_logits, selection_gold, reduction='none')
    #     loss = selection_loss.masked_select(selection_mask).sum()
    #     # TODO: mask or selection_mask ?
    #     cnt = mask.sum().item()
    #     if cnt != 0:
    #         loss /= cnt
    #     return loss

    @staticmethod
    def bce_loss(logits, labels, mask):
        selection_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = selection_loss.masked_select(mask).sum()
        cnt = mask.sum().item()
        if cnt != 0:
            loss /= cnt
        return loss

    def link_forward(self, selection_logits, element_labels, gold_labels=None):
        s_mask = element_labels == self.ss_b_id
        e_mask = torch.zeros_like(element_labels) != 0
        for e_id, e_label in enumerate(self.element_labels):
            if e_label.startswith("B-"):
                e_mask |= (element_labels == e_id)
        selection_mask = s_mask.unsqueeze(2) * e_mask.unsqueeze(1)
        selection_mask = selection_mask.unsqueeze(2).expand(-1, -1, self.num_labels, -1)

        if gold_labels is not None:
            return self.bce_loss(selection_logits, gold_labels, selection_mask)
        else:
            selection_logits = selection_logits.detach()
            selection_logits = torch.sigmoid(selection_logits) * selection_mask.float()
            selection_tags = selection_logits > 0.5
            batch_size = len(selection_tags)
            result = [[] for _ in range(batch_size)]
            idx = torch.nonzero(selection_tags.cpu())
            for i in range(idx.size(0)):
                batch, s, p, o = idx[i].tolist()
                if p == self.na_label_id:
                    continue
                rel = self.labels[p]
                result[batch].append((s, rel, o))
            return result

    @staticmethod
    def softmax_loss(inputs, labels, mask):
        num_labels = inputs.size(-1)
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

    def rel_type_forward(self, inputs, label_ids, gold_dict=None):
        batch_size, seq_len, hidden_size = inputs.shape

        batch_ss_starts = label_ids == self.ss_b_id
        batch_ss_ends = torch.zeros(batch_size, seq_len, device=batch_ss_starts.device) == 1

        start_indices = []
        for i, (batch_starts, batch_label_ids) in enumerate(zip(batch_ss_starts, label_ids)):
            j = 0
            while j < seq_len:
                if batch_starts[j]:
                    start_indices.append((i, j))
                    j += 1
                    while batch_label_ids[j] == self.ss_i_id and j < seq_len:
                        j += 1
                    batch_ss_ends[i, j - 1] = True
                else:
                    j += 1
        ss_starts_embed = inputs[batch_ss_starts]
        ss_ends_embed = inputs[batch_ss_ends]
        ss_embed = torch.cat((ss_starts_embed, ss_ends_embed), dim=-1)
        rel_mask = batch_ss_starts

        if gold_dict is not None:
            rel_loss = 0
            loss_fct = CrossEntropyLoss()
            for link_type in self.rel_labels_dict:
                gold_rel_types = gold_dict[link_type]
                hidden = self.rel_classifiers[self.rel2idx[link_type]](ss_embed)
                mask_gold_rel_types = gold_rel_types[rel_mask]
                rel_loss += loss_fct(hidden, mask_gold_rel_types)

                if torch.isnan(rel_loss):
                    exit(-2)

            return rel_loss
        else:
            rel_results = [[] for _ in range(batch_size)]
            for link_type, rel_labels in self.rel_labels_dict.items():
                hidden = self.rel_classifiers[self.rel2idx[link_type]](ss_embed)
                rel_output = torch.argmax(hidden, dim=-1)
                indices = rel_output.nonzero()
                for i in range(indices.size(0)):
                    start_idx = indices[i].tolist()[0]
                    batch, idx = start_indices[start_idx]
                    label_id = int(rel_output[start_idx])
                    rel_results[batch].append((idx, rel_labels[label_id]))

            return rel_results

    def forward(self, input_ids, features: Tuple, gold_selection_matrix=None, relative_positions=None, entity_mask=None,
                attention_mask=None, mask=None, gold_rel_type_dict=None):
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
        else:
            input_embed = self.transformer_encoder(encoder_output, mask, relative_positions=relative_positions,
                                                   entity_mask=entity_mask)[0]
            selection_logits = self.selection_layer(input_embed)

        if self.mask_entity:
            mask = features[0] != 0
        else:
            mask = mask == 1
        # TODO: 此处需注意

        if gold_selection_matrix is not None:
            link_loss = self.link_forward(selection_logits, element_labels=features[0], gold_labels=gold_selection_matrix)
            rel_loss = self.rel_type_forward(encoder_output, features[0], gold_rel_type_dict)
            return link_loss + rel_loss,
        else:
            link_pred = self.link_forward(selection_logits, element_labels=features[0])
            rel_pred = self.rel_type_forward(encoder_output, features[0])
            pred_results = [link + rel for link, rel in zip(link_pred, rel_pred)]
            return pred_results,


class SpRLModel_B_softmax(SpRLModel_B):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def link_forward(self, selection_logits, element_labels, gold_labels=None):
        s_mask = element_labels == self.ss_b_id
        e_mask = torch.zeros_like(element_labels) != 0
        for e_id, e_label in enumerate(self.element_labels):
            if e_label.startswith("B-"):
                e_mask |= (element_labels == e_id)
        selection_mask = s_mask.unsqueeze(2) * e_mask.unsqueeze(1)
        selection_mask = selection_mask == 1

        selection_logits = selection_logits.permute(0, 1, 3, 2)

        if gold_labels is not None:
            gold_labels = gold_labels.long()
            return self.softmax_loss(selection_logits, gold_labels, selection_mask)
        else:
            selection_logits = selection_logits.detach()
            selection_tags = torch.argmax(selection_logits, dim=-1)
            selection_tags[~selection_mask] = self.na_label_id
            batch_size = len(selection_tags)
            result = [[] for _ in range(batch_size)]
            idx = torch.nonzero(selection_tags != self.na_label_id)
            for i in range(idx.size(0)):
                batch, s, o = idx[i].tolist()
                p = int(selection_tags[batch, s, o])
                rel = self.labels[p]
                result[batch].append((s, rel, o))
            return result

class JointSpRLModel_B(SpRLModel_B):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.sentence_encoder.out_dim
        self.ner_layer = CRFSeqLabeling(hidden_size, self.element_labels)
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
            return element_loss + 2 * link_loss[0],
        else:
            batch_pred_element_ids = self.ner_layer(bert_output, mask=mask)
            batch_pred_element_ids[mask != 1] = self.na_element_id
            batch_pred_element_ids = batch_pred_element_ids.detach()
            batch_pred_elements = [[self.element_labels[i] for i in batch] for batch in batch_pred_element_ids]

            relative_positions_, entity_mask_ = self.get_relative_distance(batch_pred_elements, token_to_pre_idx)

            kwargs["features"] = (batch_pred_element_ids,) + kwargs["features"][1:]
            link_result = super().forward(input_ids, mask=mask, attention_mask=attention_mask,
                                          relative_positions=relative_positions_, entity_mask=entity_mask_, **kwargs)

            return [{"elements": elements, "links": links} for elements, links in
                    zip(batch_pred_elements, link_result[0])],


class JointSpRLModel_B_softmax(JointSpRLModel_B):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def link_forward(self, selection_logits, element_labels, gold_labels=None):
        s_mask = element_labels == self.ss_b_id
        e_mask = torch.zeros_like(element_labels) != 0
        for e_id, e_label in enumerate(self.element_labels):
            if e_label.startswith("B-"):
                e_mask |= (element_labels == e_id)
        selection_mask = s_mask.unsqueeze(2) * e_mask.unsqueeze(1)
        selection_mask = selection_mask == 1

        selection_logits = selection_logits.permute(0, 1, 3, 2)
        if gold_labels is not None:
            gold_labels = gold_labels.long()
            return self.softmax_loss(selection_logits, gold_labels, selection_mask)
        else:
            selection_logits = selection_logits.detach()
            selection_tags = torch.argmax(selection_logits, dim=-1)
            selection_tags[~selection_mask] = self.na_label_id
            batch_size = len(selection_tags)
            result = [[] for _ in range(batch_size)]
            idx = torch.nonzero(selection_tags != self.na_label_id)
            for i in range(idx.size(0)):
                batch, s, o = idx[i].tolist()
                p = int(selection_tags[batch, s, o])
                rel = self.labels[p]
                result[batch].append((s, rel, o))
            return result