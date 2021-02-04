import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from utils.bert_utils import bert_mask_wrapper, bert_sequence_output_wrapper
from module.crf import allowed_transitions, ConditionalRandomField
from module.lstm import LSTM
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class BertJointTokenAndSeq(BertPreTrainedModel):

    def __init__(self, config):
        super(BertJointTokenAndSeq, self).__init__(config)
        self.token_num_labels = config.token_num_labels
        self.seq_num_labels = config.seq_num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.token_classifier = nn.Linear(config.hidden_size, config.token_num_labels)
        self.seq_classifier = nn.Linear(config.hidden_size, config.seq_num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):

        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        token_logits = self.token_classifier(sequence_output)
        seq_logits = self.seq_classifier(pooled_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))

            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.token_num_labels)[active_loss]
                active_labels = token_labels.view(-1)[active_loss]
                token_loss = loss_fct(active_logits, active_labels)
            else:
                token_loss = loss_fct(token_logits.view(-1, self.token_num_labels), token_labels.view(-1))

            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits,

        return seq_logits, token_logits,


class BertCrfJointTokenAndSeq(BertJointTokenAndSeq):

    def __init__(self, config, labels=None, encoding_type=None):
        super(BertCrfJointTokenAndSeq, self).__init__(config)
        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels,
                                        encoding_type=encoding_type,
                                        include_start_end=True)
        self.crf = ConditionalRandomField(config.token_num_labels,
                                          include_start_end_trans=True,
                                          allowed_transitions=trans)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        token_logits = self.token_classifier(sequence_output)
        seq_logits = self.seq_classifier(pooled_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))
            token_loss = self.crf(token_logits, token_labels, mask).mean()
            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits
        else:
            preds, _ = self.crf.viterbi_decode(token_logits, mask)
            return seq_logits, preds,


class BertLSTMCrfJointTokenAndSeq(BertJointTokenAndSeq):

    def __init__(self, config, labels=None, encoding_type=None):
        super(BertLSTMCrfJointTokenAndSeq, self).__init__(config)
        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels,
                                        encoding_type=encoding_type,
                                        include_start_end=True)

        self.lstm_layer = LSTM(input_size=config.hidden_size,
                               hidden_size=config.hidden_size // 2,
                               num_layers=2, bidirectional=True)

        self.crf = ConditionalRandomField(config.token_num_labels,
                                          include_start_end_trans=True,
                                          allowed_transitions=trans)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        seq_logits = self.seq_classifier(pooled_output)

        sequence_output = self.dropout(sequence_output)
        batch_lengths = torch.sum(mask, -1)
        lstm_output, _ = self.lstm_layer(sequence_output, seq_len=batch_lengths)

        token_logits = self.token_classifier(lstm_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))
            token_loss = self.crf(token_logits, token_labels, mask).mean()
            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits
        else:
            preds, _ = self.crf.viterbi_decode(token_logits, mask)
            return seq_logits, preds,


class AdvBertJointTokenAndSeq(BertPreTrainedModel):

    def __init__(self, config):
        super(AdvBertJointTokenAndSeq, self).__init__(config)
        self.token_num_labels = config.token_num_labels
        self.seq_num_labels = config.seq_num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.token_classifier = nn.Linear(config.hidden_size + self.seq_num_labels, config.token_num_labels)
        self.seq_classifier = nn.Linear(config.hidden_size, config.seq_num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        seq_logits = self.seq_classifier(pooled_output)

        seq_len = token_type_ids.shape[1]
        repeat_seq_logits = seq_logits.view(-1, 1, self.seq_num_labels).repeat(1, seq_len, 1)

        sequence_output = torch.cat((sequence_output, repeat_seq_logits), -1)
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))

            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.token_num_labels)[active_loss]
                active_labels = token_labels.view(-1)[active_loss]
                token_loss = loss_fct(active_logits, active_labels)
            else:
                token_loss = loss_fct(token_logits.view(-1, self.token_num_labels), token_labels.view(-1))

            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits,

        return seq_logits, token_logits,


class AdvBertCrfJointTokenAndSeq(AdvBertJointTokenAndSeq):

    def __init__(self, config, labels=None, encoding_type=None):
        super(AdvBertCrfJointTokenAndSeq, self).__init__(config)
        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels,
                                        encoding_type=encoding_type,
                                        include_start_end=True)
        self.crf = ConditionalRandomField(config.token_num_labels,
                                          include_start_end_trans=True,
                                          allowed_transitions=trans)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        seq_logits = self.seq_classifier(pooled_output)

        seq_len = token_type_ids.shape[1]
        repeat_seq_logits = seq_logits.view(-1, 1, self.seq_num_labels).repeat(1, seq_len, 1)

        sequence_output = torch.cat((sequence_output, repeat_seq_logits), -1)
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))
            token_loss = self.crf(token_logits, token_labels, mask).mean()
            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits
        else:
            preds, _ = self.crf.viterbi_decode(token_logits, mask)
            return seq_logits, preds,


class AdvBertLSTMCrfJointTokenAndSeq(AdvBertJointTokenAndSeq):

    def __init__(self, config, labels=None, encoding_type=None):
        super(AdvBertLSTMCrfJointTokenAndSeq, self).__init__(config)
        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels,
                                        encoding_type=encoding_type,
                                        include_start_end=True)
        self.lstm_layer = LSTM(input_size=config.hidden_size,
                               hidden_size=config.hidden_size // 2,
                               num_layers=2, bidirectional=True)
        self.crf = ConditionalRandomField(config.token_num_labels,
                                          include_start_end_trans=True,
                                          allowed_transitions=trans)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):

        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        seq_logits = self.seq_classifier(pooled_output)

        seq_len = token_type_ids.shape[1]
        repeat_seq_logits = seq_logits.view(-1, 1, self.seq_num_labels).repeat(1, seq_len, 1)

        batch_lengths = torch.sum(mask, -1)
        sequence_output, _ = self.lstm_layer(sequence_output, seq_len=batch_lengths)

        sequence_output = torch.cat((sequence_output, repeat_seq_logits), -1)
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))
            token_loss = self.crf(token_logits, token_labels, mask).mean()
            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits
        else:
            preds, _ = self.crf.viterbi_decode(token_logits, mask)
            return seq_logits, preds,


class AdvBertJointTokenAndSeq_v2(BertPreTrainedModel):

    def __init__(self, config):
        super(AdvBertJointTokenAndSeq_v2, self).__init__(config)
        self.token_num_labels = config.token_num_labels
        self.seq_num_labels = config.seq_num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.token_classifier = nn.Linear(config.hidden_size * 2, config.token_num_labels)
        self.seq_classifier = nn.Linear(config.hidden_size, config.seq_num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        seq_logits = self.seq_classifier(pooled_output)

        seq_len = token_type_ids.shape[1]
        dim = pooled_output.shape[-1]
        repeat_seq_logits = pooled_output.view(-1, 1, dim).repeat(1, seq_len, 1)

        sequence_output = torch.cat((sequence_output, repeat_seq_logits), -1)
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))

            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.token_num_labels)[active_loss]
                active_labels = token_labels.view(-1)[active_loss]
                token_loss = loss_fct(active_logits, active_labels)
            else:
                token_loss = loss_fct(token_logits.view(-1, self.token_num_labels), token_labels.view(-1))

            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits,

        return seq_logits, token_logits,


class AdvBertCrfJointTokenAndSeq_v2(AdvBertJointTokenAndSeq_v2):

    def __init__(self, config, labels=None, encoding_type=None):
        super(AdvBertCrfJointTokenAndSeq_v2, self).__init__(config)
        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels,
                                        encoding_type=encoding_type,
                                        include_start_end=True)
        self.crf = ConditionalRandomField(config.token_num_labels,
                                          include_start_end_trans=True,
                                          allowed_transitions=trans)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                token_labels=None, seq_labels=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        seq_logits = self.seq_classifier(pooled_output)

        seq_len = token_type_ids.shape[1]
        dim = pooled_output.shape[-1]
        repeat_seq_logits = pooled_output.view(-1, 1, dim).repeat(1, seq_len, 1)

        sequence_output = torch.cat((sequence_output, repeat_seq_logits), -1)
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)

        if token_labels is not None and seq_labels is not None:
            loss_fct = CrossEntropyLoss()
            seq_loss = loss_fct(seq_logits.view(-1, self.seq_num_labels), seq_labels.view(-1))
            token_loss = self.crf(token_logits, token_labels, mask).mean()
            loss = seq_loss + token_loss
            return loss, seq_logits, token_logits
        else:
            preds, _ = self.crf.viterbi_decode(token_logits, mask)
            return seq_logits, preds,
