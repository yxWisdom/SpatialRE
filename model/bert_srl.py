import logging
import torch

from torch import nn
from torch.nn import CrossEntropyLoss, Parameter
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder

from utils.bert_utils import bert_sequence_output_wrapper, bert_mask_wrapper
from module.crf import allowed_transitions, ConditionalRandomField
from module.lstm import LSTM
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class EmbeddingLayer(nn.Module):
    def __init__(self, tags_num,
                 predicate_num=2,
                 input_dim=768,
                 tag_embedding_dim=128,
                 predicate_embedding_dim=128,
                 mode="concat"):
        super().__init__()
        self.tags_num = tags_num
        self.predicate_num = predicate_num
        self.tag_embedding_dim = tag_embedding_dim if mode == "concat" else input_dim
        self.predicate_embedding_dim = predicate_embedding_dim if mode == "concat" else input_dim
        self.input_dim = input_dim
        self.mode = mode

        self.predicate_embeddings = Parameter(torch.randn(predicate_num, self.predicate_embedding_dim),
                                              requires_grad=False)
        self.tag_embeddings = Parameter(torch.randn(self.tags_num, self.tag_embedding_dim), requires_grad=False)

        self.output_dim = input_dim
        if self.mode == "concat":
            self.linear = nn.Linear(self.input_dim, 512)
            # self.output_dim = self.input_dim + self.tag_embedding_dim + self.predicate_embedding_dim
        else:
            self.register_parameter('linear', None)
            # self.output_dim = self.input_dim

    def forward(self, input_layer, tag_ids, predicate_mask):
        predicate_embedding = self.predicate_embeddings[predicate_mask]
        tag_embedding = self.tag_embeddings[tag_ids]
        if self.mode == "concat":
            input_layer = self.linear(input_layer)
            output = torch.cat((input_layer, tag_embedding, predicate_embedding,), -1)
        else:
            output = input_layer + tag_embedding + predicate_embedding
        return output


# class EmbeddingLayer(nn.Module):
#     def __init__(self, tags_num,
#                  predicate_num=2,
#                  input_dim=768,
#                  tag_embedding_dim=128,
#                  predicate_embedding_dim=128,
#                  mode="concat"):
#         super().__init__()
#         self.tags_num = tags_num
#         self.predicate_num = predicate_num
#         self.tag_embedding_dim = tag_embedding_dim if mode == "concat" else input_dim
#         self.predicate_embedding_dim = predicate_embedding_dim if mode == "concat" else input_dim
#         self.input_dim = input_dim
#         self.mode = mode
#
#         self.predicate_embeddings = nn.Embedding(num_embeddings=self.predicate_num,
#                                                  embedding_dim=self.predicate_embedding_dim)
#         self.tag_embeddings = nn.Embedding(num_embeddings=self.tags_num,
#                                            embedding_dim=self.tag_embedding_dim)
#         self.predicate_embeddings.weight.requires_grad = False
#         self.tag_embeddings.weight.requires_grad = False
#         self.output_dim = self.input_dim
#
#         if self.mode == "concat":
#             self.output_dim = self.input_dim + self.tag_embedding_dim + self.predicate_embedding_dim

# def forward(self, input_layer, tag_ids, predicate_mask):
#     predicate_embedding = self.predicate_embeddings(predicate_mask)
#     tag_embedding = self.tag_embeddings(tag_ids)
#     if self.mode == "concat":
#         output = torch.cat((input_layer, tag_embedding, predicate_embedding,), -1)
#     else:
#         output = input_layer + tag_embedding + predicate_embedding
#     return output


class BertBasicSRL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tags = config.num_tags
        self.num_labels = config.num_labels

        self.embedding_layer = EmbeddingLayer(mode=config.feature_mode, tags_num=self.num_tags,
                                              input_dim=config.hidden_size)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.embedding_layer.output_dim, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, predicate_mask, tag_ids, labels=None, mask=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        sequence_output = self.embedding_layer(input_layer=sequence_output,
                                               tag_ids=tag_ids,
                                               predicate_mask=predicate_mask)
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BertLSTMSRL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tags = config.num_tags
        self.num_labels = config.num_labels

        self.embedding_layer = EmbeddingLayer(mode=config.feature_mode, tags_num=self.num_tags,
                                              input_dim=config.hidden_size)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.lstm_hidden_size = 256
        self.lstm_layer = LSTM(input_size=self.embedding_layer.output_dim,
                               hidden_size=self.lstm_hidden_size,
                               num_layers=2, bidirectional=True)

        self.classifier = nn.Linear(self.lstm_hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, predicate_mask, tag_ids, labels=None, mask=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        sequence_output = self.embedding_layer(input_layer=sequence_output,
                                               tag_ids=tag_ids,
                                               predicate_mask=predicate_mask)

        batch_lengths = torch.sum(mask, -1)
        lstm_output, _ = self.lstm_layer(sequence_output, seq_len=batch_lengths)

        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BertCrfSRL(BertPreTrainedModel):
    def __init__(self, config, labels=None, encoding_type=None):
        super().__init__(config)
        self.num_tags = config.num_tags
        self.num_labels = config.num_labels

        self.embedding_layer = EmbeddingLayer(mode=config.feature_mode, tags_num=self.num_tags,
                                              input_dim=config.hidden_size)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(self.embedding_layer.output_dim, config.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(config.num_labels, include_start_end_trans=True, allowed_transitions=trans)

        self.init_weights()

    def forward(self, input_ids, predicate_mask, tag_ids, labels=None, mask=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])

        sequence_output = self.embedding_layer(input_layer=sequence_output,
                                               tag_ids=tag_ids,
                                               predicate_mask=predicate_mask)
        sequence_output = self.dropout(sequence_output)

        logits = self.fc(sequence_output)

        if labels is None:
            preds, loss = self.crf.viterbi_decode(logits, mask)
            return preds, loss
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss,


class BertLSTMCrfSRL(BertPreTrainedModel):
    def __init__(self, config, labels=None, encoding_type=None):
        super().__init__(config)
        self.num_tags = config.num_tags
        self.num_labels = config.num_labels

        self.embedding_layer = EmbeddingLayer(mode=config.feature_mode, tags_num=self.num_tags,
                                              input_dim=config.hidden_size)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.lstm_hidden_size = 256
        self.lstm_layer = LSTM(input_size=self.embedding_layer.output_dim,
                               hidden_size=self.lstm_hidden_size,
                               num_layers=2, bidirectional=True)

        self.fc = nn.Linear(self.lstm_hidden_size * 2, config.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(config.num_labels, include_start_end_trans=True, allowed_transitions=trans)

        self.init_weights()

    def forward(self, input_ids, predicate_mask, tag_ids, labels=None, mask=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])

        sequence_output = self.embedding_layer(input_layer=sequence_output,
                                               tag_ids=tag_ids,
                                               predicate_mask=predicate_mask)

        batch_lengths = torch.sum(mask, -1)
        lstm_output, _ = self.lstm_layer(sequence_output, seq_len=batch_lengths)

        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output)

        if labels is None:
            preds, loss = self.crf.viterbi_decode(logits, mask)
            return preds, loss
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss,


class BertTransformerSRL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tags = config.num_tags
        self.num_labels = config.num_labels

        self.embedding_layer = EmbeddingLayer(mode=config.feature_mode, tags_num=self.num_tags,
                                              input_dim=config.hidden_size)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        encoder_layer = TransformerEncoderLayer(d_model=self.embedding_layer.output_dim,
                                                nhead=self.embedding_layer.output_dim // 64,
                                                dim_feedforward=config.intermediate_size,
                                                dropout=config.hidden_dropout_prob,
                                                activation="gelu")

        self.transformer_encoder = TransformerEncoder(encoder_layer, 6)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, predicate_mask, tag_ids, labels=None, mask=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        sequence_output = self.embedding_layer(input_layer=sequence_output,
                                               tag_ids=tag_ids,
                                               predicate_mask=predicate_mask)
        transformer_mask = mask == 0
        transformer_output = self.transformer_encoder(sequence_output.transpose(0, 1),
                                                      src_key_padding_mask=transformer_mask).transpose(0, 1)
        logits = self.classifier(transformer_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BertTransformerCrfSRL(BertPreTrainedModel):
    def __init__(self, config, labels=None, encoding_type=None):
        super().__init__(config)
        self.num_tags = config.num_tags
        self.num_labels = config.num_labels

        self.embedding_layer = EmbeddingLayer(mode=config.feature_mode, tags_num=self.num_tags,
                                              input_dim=config.hidden_size)

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # for name, param in self.bert.named_parameters():
        #     if '11' in name or '10' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        encoder_layer = TransformerEncoderLayer(d_model=self.embedding_layer.output_dim,
                                                nhead=self.embedding_layer.output_dim // 128,
                                                dim_feedforward=config.intermediate_size,
                                                dropout=config.hidden_dropout_prob,
                                                activation="gelu")

        self.transformer_encoder = TransformerEncoder(encoder_layer, 6)

        self.fc = nn.Linear(config.hidden_size, config.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(config.num_labels, include_start_end_trans=True, allowed_transitions=trans)

        self.init_weights()

    def forward(self, input_ids, predicate_mask, tag_ids, labels=None, mask=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])

        sequence_output = self.embedding_layer(input_layer=sequence_output,
                                               tag_ids=tag_ids,
                                               predicate_mask=predicate_mask)
        transformer_mask = mask == 0
        transformer_output = self.transformer_encoder(sequence_output.transpose(0, 1),
                                                      src_key_padding_mask=transformer_mask).transpose(0, 1)

        logits = self.fc(transformer_output)

        if labels is None:
            preds, loss = self.crf.viterbi_decode(logits, mask)
            return preds, loss
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss,
# class BertBasicSRL(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#
#         self.num_tags = config.num_tags
#         self.num_labels = config.num_labels
#         self.predicate_dim = 256
#         self.predicate_embeddings = Parameter(torch.randn(2, self.predicate_dim), requires_grad=False)
#
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size + self.predicate_dim, config.num_labels)
#
#         self.init_weights()
#
#     def forward(self, input_ids, predicate_mask, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None, labels=None):
#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask)
#
#         sequence_output = outputs[0]
#         predicate_embedding = self.predicate_embeddings[predicate_mask]
#         # predicate_embedding = self.predicate_embeddings(predicate_mask)
#
#         sequence_output = torch.cat((sequence_output, predicate_embedding), -1)
#         sequence_output = self.dropout(sequence_output)
#
#         logits = self.classifier(sequence_output)
#
#         outputs = (logits,)
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)[active_loss]
#                 active_labels = labels.view(-1)[active_loss]
#                 loss = loss_fct(active_logits, active_labels)
#             else:
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs
#
#         return outputs
#
#
# class BertCrfSRL(BertPreTrainedModel):
#     def __init__(self, config, labels=None, encoding_type=None):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.predicate_dim = 256
#         # self.predicate_embeddings = nn.Embedding(2, self.predicate_dim)
#         self.predicate_embeddings = Parameter(torch.randn(2, self.predicate_dim), requires_grad=False)
#
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.fc = nn.Linear(config.hidden_size + self.predicate_dim, config.num_labels)
#
#         trans = None
#         if labels is not None and encoding_type is not None:
#             id2labels = {idx: label for idx, label in enumerate(labels)}
#             trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
#         self.crf = ConditionalRandomField(config.num_labels, include_start_end_trans=True, allowed_transitions=trans)
#
#         self.init_weights()
#
#     def forward(self, input_ids, predicate_mask, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None, labels=None):
#
#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask)
#
#         sequence_output = outputs[0]
#         predicate_embedding = self.predicate_embeddings[predicate_mask]
#
#         sequence_output = torch.cat((sequence_output, predicate_embedding), -1)
#         sequence_output = self.dropout(sequence_output)
#
#         logits = self.fc(sequence_output)
#
#         if labels is None:
#             preds, loss = self.crf.viterbi_decode(logits, attention_mask)
#             return preds, loss
#         else:
#             loss = self.crf(logits, labels, attention_mask).mean()
#             return loss,
