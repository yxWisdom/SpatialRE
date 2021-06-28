import logging

import torch

from utils.bert_utils import bert_sequence_output_wrapper, bert_mask_wrapper
from pytorch_transformers import RobertaModel
from torch import nn
from torch.nn import CrossEntropyLoss

from module.crf import allowed_transitions, ConditionalRandomField
from module.lstm import LSTM
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class BasicBertSeqLabeling(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """

    def __init__(self, config):
        super(BasicBertSeqLabeling, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, labels=None, mask=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):

        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
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

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrf(BertPreTrainedModel):
    def __init__(self, config, labels=None, encoding_type=None):
        super(BertCrf, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(config.num_labels, include_start_end_trans=True, allowed_transitions=trans)

        self.init_weights()

    def forward(self, input_ids, labels=None, mask=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):

        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])

        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)

        if labels is None:
            preds, loss = self.crf.viterbi_decode(logits, mask)
            return preds, loss
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss,


class BertLSTMCrf(BertPreTrainedModel):
    def __init__(self, config, labels=None, encoding_type=None):
        super(BertLSTMCrf, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.lstm_layer = LSTM(input_size=config.hidden_size,
                               hidden_size=config.hidden_size,
                               num_layers=2, bidirectional=True)

        self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(config.num_labels, include_start_end_trans=True, allowed_transitions=trans)

        self.init_weights()

    def forward(self, input_ids, labels=None, mask=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):

        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])

        batch_lengths = torch.sum(mask, -1)
        lstm_output, _ = self.lstm_layer(sequence_output, seq_len=batch_lengths)

        # sequence_output = self.dropout(sequence_output)
        logits = self.fc(lstm_output)

        if labels is None:
            preds, loss = self.crf.viterbi_decode(logits, mask)
            return preds, loss
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss,


class RobertaCrf(BertPreTrainedModel):

    def __init__(self, config, labels=None, encoding_type=None):
        super(RobertaCrf, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(config.num_labels, include_start_end_trans=True, allowed_transitions=trans)

        self.init_weights()

    def forward(self, input_ids, labels=None, mask=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        sequence_output = bert_sequence_output_wrapper(outputs[0])

        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)

        if labels is None:
            preds, loss = self.crf.viterbi_decode(logits, mask)
            return preds, loss
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss,


class CRFSeqLabeling(nn.Module):
    def __init__(self, hidden_size, labels):
        super().__init__()
        self.labels = labels
        self.num_labels = len(self.labels)
        self.fc = nn.Linear(hidden_size, self.num_labels)
        id2labels = {idx: label for idx, label in enumerate(labels)}
        trans = allowed_transitions(id2labels, include_start_end=True)
        self.crf = ConditionalRandomField(self.num_labels, include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, inputs, labels=None, mask=None):
        logits = self.fc(inputs)
        if labels is None:
            pred_labels, _ = self.crf.viterbi_decode(logits, mask)
            return pred_labels
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss


class LSTMCRFSeqLabeling(nn.Module):
    def __init__(self, hidden_size, labels):
        super().__init__()
        self.labels = labels
        self.num_labels = len(self.labels)
        self.bi_lstm = LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, self.num_labels)

        id2labels = {idx: label for idx, label in enumerate(labels)}
        trans = allowed_transitions(id2labels, include_start_end=True)
        self.crf = ConditionalRandomField(self.num_labels, include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, inputs, labels=None, mask=None):
        seq_lengths = torch.sum(mask, -1)
        output, _ = self.bi_lstm(inputs, seq_len=seq_lengths)
        logits = self.fc(output)
        if labels is None:
            pred_labels, _ = self.crf.viterbi_decode(logits, mask)
            return pred_labels
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss

