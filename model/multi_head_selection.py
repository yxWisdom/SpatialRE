# import torch
# from torch.nn import init
#
# from utils.bert_utils import bert_mask_wrapper, bert_sequence_output_wrapper
# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# from torch import nn
# import torch.nn.functional as F
#
# from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
#
#
# # class MHSEmbeddingLayer(nn.Module):
# #     def __init__(self):
#
#
# class MultiHeadCell(nn.Module):
#     def __init__(self, input_size, hidden_size, relation_num):
#         super(MultiHeadCell, self).__init__()
#         # self.selection_u = nn.Linear(input_size, self.hidden_size)
#         # self.selection_v = nn.Linear(input_size, self.hidden_size)
#
#         self.selection_u = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=True)
#         self.selection_v = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=True)
#
#         self.v = nn.Parameter(torch.randn(hidden_size, relation_num), requires_grad=True)
#         self.bias = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
#
#         self.init_weights()
#
#     def init_weights(self):
#         init.kaiming_uniform_(self.v)
#         init.kaiming_uniform_(self.selection_u)
#         init.kaiming_uniform_(self.selection_v)
#         init.uniform_(self.bias, -1, 1)
#
#     @staticmethod
#     def broadcasting(left, right):
#         x2 = left.permute(1, 0, 2).unsqueeze(-1)
#         y2 = right.permute(0, 2, 1).unsqueeze(0)
#         return (x2 + y2).permute(1, 0, 3, 2)
#
#     def forward(self, _input):
#         left = torch.einsum('aij,jk->aik', _input, self.selection_u)
#         right = torch.einsum('aij,jk->aik', _input, self.selection_v)
#
#         outer_sum = self.broadcasting(left, right)
#         outer_sum_bias = outer_sum + self.bias
#
#         output = torch.einsum('bijh,hr->birj', outer_sum_bias, self.v)
#
#         return output
#
#
# class BertMultiHeadSelection(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#
#         self.num_labels = len(config.labels)
#         self.num_relations = len(config.relations)
#
#         self.relations = config.relations
#         self.na_relation = self.relations.index("NA")
#
#         self.relation_embedding_dim = 256
#
#         self.label_embedding = nn.Embedding(num_embeddings=self.num_labels, embedding_dim=config.hidden_size)
#         self.relation_embedding = nn.Embedding(num_embeddings=self.num_relations, embedding_dim=config.hidden_size)
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.bert = BertModel(config)
#
#         self.selection_cell = MultiHeadCell(config.hidden_size, 256, self.num_relations)
#         self.init_weights()
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
#         sequence_output = sequence_output + self.label_embedding(labels)
#         sequence_output = self.dropout(sequence_output)
#
#         selection_logits = self.selection_cell(sequence_output)
#
#         if gold_selection_matrix is not None:
#             loss = self.masked_loss(selection_logits, gold_selection_matrix, mask)
#             return loss,
#         else:
#             return self.inference(selection_logits, mask),
#
#
# class TransformerMultiHeadSelection(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#
#         self.num_labels = len(config.labels)
#         self.num_relations = len(config.relations)
#
#         self.relations = config.relations
#         self.na_relation = self.relations.index("NA")
#
#         self.relation_embedding_dim = 256
#
#         self.label_embedding = nn.Embedding(num_embeddings=self.num_labels, embedding_dim=config.hidden_size)
#         self.relation_embedding = nn.Embedding(num_embeddings=self.num_relations, embedding_dim=config.hidden_size)
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.bert = BertModel(config)
#
#         # for name, param in self.bert.named_parameters():
#         #     if '11' in name:
#         #         param.requires_grad = True
#         #     else:
#         #         param.requires_grad = False
#
#         encoder_layer = TransformerEncoderLayer(config.hidden_size, 12, config.intermediate_size,
#                                                 config.hidden_dropout_prob)
#
#         self.transformer_encoder = TransformerEncoder(encoder_layer, 12)
#
#         self.selection_cell = MultiHeadCell(config.hidden_size, 256, self.num_relations)
#         self.init_weights()
#
#         print()
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
#
#     def forward(self, input_ids, labels, gold_selection_matrix=None, mask=None, attention_mask=None,
#                 token_type_ids=None, position_ids=None, head_mask=None):
#         if mask is None and attention_mask is not None:
#             mask = bert_mask_wrapper(attention_mask)
#
#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask)
#
#         sequence_output = bert_sequence_output_wrapper(outputs[0])
#         sequence_output = sequence_output + self.label_embedding(labels)
#         # sequence_output = self.dropout(sequence_output)
#
#         transformer_mask = mask == 0
#         transformer_output = self.transformer_encoder(sequence_output.transpose(0, 1),
#                                                       src_key_padding_mask=transformer_mask).transpose(0, 1)
#
#         selection_logits = self.selection_cell(transformer_output)
#
#         if gold_selection_matrix is not None:
#             mask = mask == 1
#             loss = self.masked_loss(selection_logits, gold_selection_matrix, mask=mask)
#             return loss,
#         else:
#             return self.inference(selection_logits, mask),
