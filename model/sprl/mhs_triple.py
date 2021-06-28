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
from allennlp.nn.util import batched_index_select

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
        self.rel_labels_dict_rev = {
            link_type: {label: idx for idx, label in enumerate(labels)}
            for link_type, labels in self.rel_labels_dict.items()
        }
        rel_types = list(sorted(self.rel_labels_dict.keys()))
        self.rel2idx = {rel: idx for idx, rel in enumerate(rel_types)}

        self.ss_b_id = self.element_labels.index("B-SPATIAL_SIGNAL")
        self.ss_i_id = self.element_labels.index("I-SPATIAL_SIGNAL")
        hidden_size = self.sentence_encoder.out_dim

        self.rel_classifiers = nn.ModuleList([
            nn.Sequential(
                FeedForward(input_dim=hidden_size * 3,
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

    def link_forward(self, selection_logits, element_labels, mask, gold_labels=None):
        s_mask = element_labels == self.ss_b_id
        e_mask = mask
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

    def get_element_ids(self, element_ids, tags, elements, token_to_pre_idx):
        seq_len = len(tags)
        if seq_len != 64:
            print(1)
        try:
            modified_element_ids = [element_ids[idx] if idx < seq_len else '' for idx in token_to_pre_idx]
        except IndexError:
            print(1)
        new_element_ids = [x for x in modified_element_ids]
        for idx in elements:
            if modified_element_ids[idx] == "":
                new_element_ids[idx] = f"na-{token_to_pre_idx[idx]}"
        return new_element_ids

    def get_triples(self, batch_pred_triples, batch_element_ids, batch_tag_ids, batch_token_to_pre_idx):
        batch_size = len(batch_pred_triples)
        batch_id_triples = [[] for _ in range(batch_size)]
        batch_idx_triples = [[] for _ in range(batch_size)]
        batch_trigger_dict = [defaultdict(lambda: defaultdict(set)) for _ in range(batch_size)]
        batch_elements = [set() for _ in range(batch_size)]

        for batch, triples in enumerate(batch_pred_triples):
            for (s, rel, o) in triples:
                batch_trigger_dict[batch][s][rel].add(o)
                batch_elements[batch].add(s)
                batch_elements[batch].add(o)

        max_relation_num = 0
        for i, trigger2roles in enumerate(batch_trigger_dict):
            triple_set = set()

            element_ids = batch_element_ids[i]
            token_to_pre_idx = batch_token_to_pre_idx[i]
            tag_ids = batch_tag_ids[i]
            tags = [self.element_labels[int(tag)] for tag in tag_ids]
            element_ids = self.get_element_ids(element_ids, tags, batch_elements[i], token_to_pre_idx)
            for trigger, role_dict in trigger2roles.items():
                # TODO:
                role_dict["landmark"].add('')

                sp_id = element_ids[trigger]
                if tags[trigger] != "B-SPATIAL_SIGNAL":
                    continue
                for tr in role_dict.get("trajector", {''}):
                    tr_id = element_ids[tr] if tr != '' else ''
                    if tr_id == sp_id:
                        continue
                    # if tr == '' or tags[tr] != "B-SPATIAL_ENTITY":
                    if tr != '' and tags[tr] != "B-SPATIAL_ENTITY":
                        continue
                    for ld in role_dict.get("landmark", {''}):
                        ld_id = element_ids[ld] if ld != '' else ''
                        if tr_id == ld_id or sp_id == ld_id:
                            continue
                        # if ld == '' or tags[ld] != "B-SPATIAL_ENTITY":
                        if ld != '' and tags[ld] != "B-SPATIAL_ENTITY":
                            continue
                        id_triple = (sp_id, tr_id, ld_id)
                        if id_triple in triple_set:
                            continue
                        triple_set.add(id_triple)
                        batch_idx_triples[i].append((trigger, tr, ld))
                        batch_id_triples[i].append(id_triple)

            max_relation_num = max(max_relation_num, len(batch_idx_triples[i]))

        return batch_idx_triples, batch_id_triples, max_relation_num

    def relation_forward(self, sequence_embed, pred_triples, element_ids, tag_ids, token_to_pre_idx,
                         gold_relations=None, max_relation_num=24):
        batch_idx_triples, batch_id_triples, max_relation_num = self.get_triples(pred_triples, element_ids,
                                                                                 tag_ids, token_to_pre_idx)
        batch_relation_triples = []
        batch_relation_masks = []
        batch_relation_labels = {link_type: [] for link_type in self.rel_labels_dict}

        batch_size, seq_len, hidden_size = sequence_embed.shape
        pad_idx = seq_len


        if max_relation_num == 0:
            if gold_relations is not None:
                return 0
            else:
                return [[] for _ in range(batch_size)]

        for batch, (idx_triples, id_triples) in enumerate(zip(batch_idx_triples, batch_id_triples)):
            relation_label_dict = {link_type: [] for link_type in self.rel_labels_dict} #defaultdict(list)
            relation_triples = []
            for idx_triple, id_triple in zip(idx_triples, id_triples):
                if gold_relations is not None:
                    for link_type in self.rel_labels_dict:
                        gold_relation_dict = gold_relations[batch].get(link_type, dict())
                        # for link_type, relation_dict in gold_relations[batch].items():
                        if id_triple in gold_relation_dict:
                            rel_label = gold_relation_dict[id_triple]
                            rel_label_id = self.rel_labels_dict_rev[link_type][rel_label]
                            relation_label_dict[link_type].append(rel_label_id)
                        else:
                            relation_label_dict[link_type].append(0)

                idx_triple = list(map(lambda x: pad_idx if x == '' else x, idx_triple))
                relation_triples.append(idx_triple)

            rel_num = len(idx_triples)
            pad_len = max_relation_num - rel_num
            relation_mask = [1] * rel_num + [0] * pad_len
            relation_triples = relation_triples + [[pad_idx, pad_idx, pad_idx] for _ in range(pad_len)]

            if gold_relations is not None:
                for link_type, relation_labels in relation_label_dict.items():
                    batch_relation_labels[link_type].append(relation_labels + [0] * pad_len)

            batch_relation_masks.append(relation_mask)
            batch_relation_triples.append(relation_triples)

        device = sequence_embed.device
        relation_triples_tensor = torch.tensor(batch_relation_triples, dtype=torch.long, device=device)
        relation_masks_tensor = torch.tensor(batch_relation_masks, dtype=torch.long, device=device) == 1
        # relation_labels_tensor = torch.tensor(batch_relation_labels, dtype=torch.long, device=device)

        sps = relation_triples_tensor[:, :, 0].view(batch_size, -1)
        trs = relation_triples_tensor[:, :, 1].view(batch_size, -1)
        lds = relation_triples_tensor[:, :, 2].view(batch_size, -1)

        zero_embed = torch.zeros((batch_size, 1, hidden_size), dtype=torch.float, device=device)
        sequence_embed = torch.cat((sequence_embed, zero_embed), dim=1)

        sp_embed = batched_index_select(sequence_embed, sps)
        tr_embed = batched_index_select(sequence_embed, trs)
        ld_embed = batched_index_select(sequence_embed, lds)

        relation_embed = torch.cat((tr_embed, sp_embed, ld_embed), dim=-1)

        if gold_relations is not None:
            loss = 0
            for link_type, classifier_id in self.rel2idx.items():
                rel_logits = self.rel_classifiers[classifier_id](relation_embed)
                gold_rel_labels = torch.tensor(batch_relation_labels[link_type], dtype=torch.long, device=device)
                rel_loss = self.softmax_loss(rel_logits, gold_rel_labels, relation_masks_tensor)
                loss += rel_loss
            return loss

        else:
            result = [[] for _ in range(batch_size)]
            for link_type, classifier_id in self.rel2idx.items():
                rel_logits = self.rel_classifiers[classifier_id](relation_embed)
                rel_output = torch.argmax(rel_logits, dim=-1)
                for batch, id_triples in enumerate(batch_id_triples):
                    for idx, id_triple in enumerate(id_triples):
                        rel_id = rel_output[batch, idx]
                        if rel_id != 0:
                            rel_label = self.rel_labels_dict[link_type][rel_id]
                            result[batch].append(id_triple + (rel_label,))
            return result

    def forward(self, input_ids, features: Tuple, gold_selection_matrix=None, relative_positions=None, entity_mask=None,
                attention_mask=None, mask=None, element_ids=None, token_to_pre_idx=None, gold_relations=None):
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
            link_loss = self.link_forward(selection_logits, features[0], mask, gold_labels=gold_selection_matrix)
            triples = self.link_forward(selection_logits, features[0], mask)
            rel_loss = self.relation_forward(encoder_output, triples, element_ids, features[0],
                                             token_to_pre_idx, gold_relations)
            return link_loss + rel_loss,
        else:
            triples = self.link_forward(selection_logits, features[0], mask)
            rel_pred = self.relation_forward(encoder_output, triples, element_ids, features[0],
                                             token_to_pre_idx)
            return rel_pred,

class SpRLModel_B_softmax(SpRLModel_B):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def link_forward(self, selection_logits, element_labels, mask, gold_labels=None):
        s_mask = element_labels == self.ss_b_id
        e_mask = mask
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
                                        token_to_pre_idx=token_to_pre_idx,
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
                                          token_to_pre_idx=token_to_pre_idx,
                                          relative_positions=relative_positions_, entity_mask=entity_mask_, **kwargs)

            return [{"elements": elements, "links": links} for elements, links in
                    zip(batch_pred_elements, link_result[0])],


class JointSpRLModel_B_softmax(JointSpRLModel_B):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def link_forward(self, selection_logits, element_labels, mask, gold_labels=None):
        s_mask = element_labels == self.ss_b_id
        e_mask = mask
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