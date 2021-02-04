from typing import List, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from module.entity_aware_transformer import EATransformerEncoderLayer, EATransformerEncoder
from utils.bert_utils import bert_mask_wrapper, bert_sequence_output_wrapper
from module.crf import allowed_transitions, ConditionalRandomField
from module.lstm import LSTM
from module.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules import TransformerEncoderLayer as TfEncoderLayer, TransformerEncoder as TfEncoder
from pytorch_transformers import BertModel


class BertSRLEncoder(nn.Module):
    def __init__(self, pretrain_path, out_dim, features: List[int], bert_in_dim=768, bert_out_dim=256, feature_dim=64,
                 feature_mode="concat", custom_embed=True, embed_requires_grad=False, use_pos_embed=False,
                 max_seq_len=512,
                 fine_tuning=True):
        super().__init__()
        self.feature_mode = feature_mode
        self.out_dim = out_dim
        self.custom_embed = custom_embed
        self.use_pos_embed = use_pos_embed

        # features_num, features_dim = zip(*features)
        # features_size = reduce(lambda x, y: x + y, features_dim)

        # if use position embedding, add maximum sequence length to feature list
        if use_pos_embed:
            features.append(max_seq_len)

        total_feature_dim = len(features) * feature_dim

        if self.feature_mode == "concat" and out_dim != total_feature_dim + bert_out_dim:
            raise ValueError("features size is not same as out_dim")

        if self.feature_mode != "concat" and out_dim != bert_out_dim:
            raise ValueError("bert_out_dim is not same as out_dim")

        self.feature_dim = feature_dim if self.feature_mode == "concat" else out_dim

        if bert_out_dim != bert_in_dim:
            self.dense = nn.Linear(bert_in_dim, bert_out_dim)

        layers = []

        for feature_num in features:
            if custom_embed:
                embed_layer = nn.Parameter(torch.randn(feature_num, self.feature_dim),
                                           requires_grad=embed_requires_grad)
            else:
                embed_layer = nn.Embedding(feature_num, self.feature_dim)
                embed_layer.weight.requires_grad = embed_requires_grad
            layers.append(embed_layer)

        self.feature_embed_layers = nn.ParameterList(layers) if custom_embed else nn.ModuleList(layers)

        self.bert = BertModel.from_pretrained(pretrain_path)

        if not fine_tuning:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False

    def forward(self, tokens, features: Tuple, attention_mask=None):
        sequence_output = self.bert(tokens, attention_mask=attention_mask)[0]
        if hasattr(self, "dense"):
            sequence_output = self.dense(sequence_output)
        sequence_output = bert_sequence_output_wrapper(sequence_output)

        if self.use_pos_embed:
            seq_length = tokens.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand_as(tokens)
            features += (position_ids,)

        feature_embeds = ()
        for layer, feature in zip(self.feature_embed_layers, features):
            if self.custom_embed:
                feature_embeds += (layer[feature],)
            else:
                feature_embeds += (layer(feature),)

        if self.feature_mode == "concat":
            feature_embeds = (sequence_output,) + feature_embeds
            output = torch.cat(feature_embeds, -1)
        else:
            output = sequence_output
            for feature_embed in feature_embeds:
                output += feature_embed

        return output


class LstmSoftmaxSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, lstm_hidden_size=256, dropout=0.1, num_layers=2):
        super().__init__()
        self.encoder_layer = sentence_encoder
        # self.id2labels = {idx: label for idx, label in enumerate(labels)}
        # self.label2ids = {label: idx for idx, label in enumerate(labels)}
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = LSTM(input_size=self.encoder_layer.out_dim,
                         hidden_size=self.lstm_hidden_size,
                         num_layers=num_layers, bidirectional=True)

        self.classifier = nn.Linear(self.lstm_hidden_size * 2, self.num_labels)

    def forward(self, input_ids, features: Tuple, labels=None, attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        batch_lengths = torch.sum(mask, -1)
        lstm_output, _ = self.lstm(encoder_output, seq_len=batch_lengths)

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
        else:
            pred = torch.argmax(logits, -1)
            outputs = (pred,) + outputs

        return outputs


class LstmCrfSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, lstm_hidden_size=256, dropout=0.1, num_layers=2, encoding_type="BIO"):
        super().__init__()
        self.encoder_layer = sentence_encoder
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = LSTM(input_size=self.encoder_layer.out_dim,
                         hidden_size=self.lstm_hidden_size,
                         num_layers=num_layers, bidirectional=True)

        self.fc = nn.Linear(self.lstm_hidden_size * 2, self.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(self.num_labels, include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, input_ids, features: Tuple, labels=None, attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        batch_lengths = torch.sum(mask, -1)
        lstm_output, _ = self.lstm(encoder_output, seq_len=batch_lengths)

        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output)

        if labels is None:
            pred, scores = self.crf.viterbi_decode(logits, mask)
            return pred, logits, scores
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss, logits


class TransformerSoftmaxSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, dropout=0.1,
                 attn_dropout=0.1, activation="relu"):
        super().__init__()
        self.encoder_layer = sentence_encoder
        # self.id2labels = {idx: label for idx, label in enumerate(labels)}
        # self.label2ids = {label: idx for idx, label in enumerate(labels)}
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)

        hidden_size = self.encoder_layer.out_dim

        encoder_layer = TfEncoderLayer(d_model=hidden_size, nhead=num_attention_heads,
                                       dim_feedforward=dim_feedforward, dropout=attn_dropout, activation=activation)
        self.transformer_encoder = TfEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, input_ids, features: Tuple, labels=None, attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        transformer_mask = mask == 0
        transformer_output = self.transformer_encoder(encoder_output.transpose(0, 1),
                                                      src_key_padding_mask=transformer_mask).transpose(0, 1)

        transformer_output = self.dropout(transformer_output)
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
        else:
            pred = torch.argmax(logits, -1)
            outputs = (pred,) + outputs

        return outputs


class TransformerCrfSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, dropout=0.1,
                 attn_dropout=0.1, activation="relu", encoding_type="BIO"):
        super().__init__()
        self.encoder_layer = sentence_encoder
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)

        hidden_size = self.encoder_layer.out_dim

        encoder_layer = TfEncoderLayer(d_model=hidden_size, nhead=num_attention_heads,
                                       dim_feedforward=dim_feedforward, dropout=attn_dropout, activation=activation)
        self.transformer_encoder = TfEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(hidden_size, self.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(self.num_labels, include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, input_ids, features: Tuple, labels=None, attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        transformer_mask = mask == 0
        transformer_output = self.transformer_encoder(encoder_output.transpose(0, 1),
                                                      src_key_padding_mask=transformer_mask).transpose(0, 1)

        output = self.dropout(transformer_output)
        logits = self.fc(output)

        if labels is None:
            pred, scores = self.crf.viterbi_decode(logits, mask)
            return pred, logits, scores
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss, logits


class MyTransformerSoftmaxSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, dropout=0.1,
                 attn_dropout=0.1, activation="relu"):
        super().__init__()
        self.encoder_layer = sentence_encoder
        # self.id2labels = {idx: label for idx, label in enumerate(labels)}
        # self.label2ids = {label: idx for idx, label in enumerate(labels)}
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)

        hidden_size = self.encoder_layer.out_dim

        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, n_head=num_attention_heads,
                                                dim_feedforward=dim_feedforward, dropout=attn_dropout,
                                                activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, input_ids, features: Tuple, labels=None, attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        output = self.transformer_encoder(encoder_output, mask)[0]
        output = self.dropout(output)
        logits = self.classifier(output)

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
        else:
            pred = torch.argmax(logits, -1)
            outputs = (pred,) + outputs

        return outputs


class MyTransformerCrfSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, dropout=0.1,
                 attn_dropout=0.1, activation="relu", encoding_type="BIO"):
        super().__init__()
        self.encoder_layer = sentence_encoder
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)

        hidden_size = self.encoder_layer.out_dim
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, n_head=num_attention_heads,
                                                dim_feedforward=dim_feedforward, dropout=attn_dropout,
                                                activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(hidden_size, self.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(self.num_labels, include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, input_ids, features: Tuple, labels=None, attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        output = self.transformer_encoder(encoder_output, mask)[0]
        output = self.dropout(output)
        logits = self.fc(output)

        if labels is None:
            pred, scores = self.crf.viterbi_decode(logits, mask)
            return pred, logits, scores
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss, logits


class EntityAwareTransformerSoftmaxSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, dropout=0.1,
                 attn_dropout=0.1, activation="relu", max_distance=2):
        super().__init__()
        self.encoder_layer = sentence_encoder
        # self.id2labels = {idx: label for idx, label in enumerate(labels)}
        # self.label2ids = {label: idx for idx, label in enumerate(labels)}
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)

        hidden_size = self.encoder_layer.out_dim

        encoder_layer = EATransformerEncoderLayer(d_model=hidden_size, n_head=num_attention_heads,
                                                  dim_feedforward=dim_feedforward, dropout=attn_dropout,
                                                  activation=activation)
        self.transformer_encoder = EATransformerEncoder(encoder_layer, num_layers,
                                                        hidden_size=hidden_size,
                                                        max_distance=max_distance)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, input_ids, features: Tuple, relative_positions=None, entity_mask=None, labels=None,
                attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
            mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        output = self.transformer_encoder(encoder_output, mask,
                                          relative_positions=relative_positions,
                                          entity_mask=entity_mask)[0]
        output = self.dropout(output)
        logits = self.classifier(output)

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
        else:
            pred = torch.argmax(logits, -1)
            outputs = (pred,) + outputs

        return outputs


class EntityAwareTransformerCrfSRL(nn.Module):
    def __init__(self, sentence_encoder, labels, num_layers, num_attention_heads, dim_feedforward, dropout=0.1,
                 attn_dropout=0.1, activation="relu", encoding_type="BIO", max_distance=2):
        super().__init__()
        self.encoder_layer = sentence_encoder
        self.num_labels = len(labels)

        self.dropout = nn.Dropout(dropout)

        hidden_size = self.encoder_layer.out_dim
        encoder_layer = EATransformerEncoderLayer(d_model=hidden_size, n_head=num_attention_heads,
                                                  dim_feedforward=dim_feedforward, dropout=attn_dropout,
                                                  activation=activation)
        self.transformer_encoder = EATransformerEncoder(encoder_layer, num_layers,
                                                        hidden_size=hidden_size,
                                                        max_distance=max_distance)

        self.fc = nn.Linear(hidden_size, self.num_labels)

        trans = None
        if labels is not None and encoding_type is not None:
            id2labels = {idx: label for idx, label in enumerate(labels)}
            trans = allowed_transitions(id2labels, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(self.num_labels, include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, input_ids, features: Tuple, relative_positions=None, entity_mask=None, labels=None,
                attention_mask=None, mask=None):
        if mask is None and attention_mask is not None:
                mask = bert_mask_wrapper(attention_mask)

        encoder_output = self.encoder_layer(input_ids, features, attention_mask)

        output = self.transformer_encoder(encoder_output, mask,
                                          relative_positions=relative_positions,
                                          entity_mask=entity_mask)[0]
        output = self.dropout(output)
        logits = self.fc(output)

        if labels is None:
            pred, scores = self.crf.viterbi_decode(logits, mask)
            return pred, logits, scores
        else:
            loss = self.crf(logits, labels, mask).mean()
            return loss, logits
