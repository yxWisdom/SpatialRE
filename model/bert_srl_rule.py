import logging
import torch

from torch import nn
from torch.nn import Parameter
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder

from utils.bert_utils import bert_sequence_output_wrapper, bert_mask_wrapper
from module.crf import allowed_transitions, ConditionalRandomField
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class EmbeddingLayer(nn.Module):
    def __init__(self, tags_num,
                 pretag_num,
                 predicate_num=2,
                 input_dim=768,
                 tag_embedding_dim=128,
                 predicate_embedding_dim=128,
                 pretag_embedding_dim=128,
                 mode="concat"):
        super().__init__()
        self.tags_num = tags_num
        self.predicate_num = predicate_num
        self.pretag_num = pretag_num

        self.tag_embedding_dim = tag_embedding_dim if mode == "concat" else input_dim
        self.predicate_embedding_dim = predicate_embedding_dim if mode == "concat" else input_dim
        self.pretag_embedding_dim = pretag_embedding_dim if mode == "concat" else input_dim
        self.input_dim = input_dim
        self.mode = mode

        self.predicate_embeddings = Parameter(torch.randn(predicate_num, self.predicate_embedding_dim),
                                              requires_grad=False)
        self.tag_embeddings = Parameter(torch.randn(self.tags_num, self.tag_embedding_dim), requires_grad=False)
        self.pretag_embeddings = Parameter(torch.randn(self.pretag_num, self.pretag_embedding_dim), requires_grad=False)

        self.output_dim = input_dim
        if self.mode == "concat":
            self.linear = nn.Linear(self.input_dim, 384)
            # self.output_dim = self.input_dim + self.tag_embedding_dim + self.predicate_embedding_dim
        else:
            self.register_parameter('linear', None)
            # self.output_dim = self.input_dim

    def forward(self, input_layer, tag_ids, pretag_ids, predicate_mask):
        predicate_embedding = self.predicate_embeddings[predicate_mask]
        tag_embedding = self.tag_embeddings[tag_ids]
        pretag_embedding = self.pretag_embeddings[pretag_ids]
        if self.mode == "concat":
            input_layer = self.linear(input_layer)
            output = torch.cat((input_layer, tag_embedding, predicate_embedding, pretag_embedding), -1)
        else:
            output = input_layer + tag_embedding + predicate_embedding + pretag_embedding
        return output


class BertTransformerCrfWithRuleSRL(BertPreTrainedModel):
    def __init__(self, config, labels=None, encoding_type=None):
        super().__init__(config)
        self.num_tags = config.num_tags
        self.num_labels = config.num_labels

        self.embedding_layer = EmbeddingLayer(mode=config.feature_mode, tags_num=self.num_tags, pretag_num=self.num_labels,
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

    def forward(self, input_ids, predicate_mask, tag_ids, pretag_ids, labels=None, mask=None, attention_mask=None,
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
                                               pretag_ids=pretag_ids,
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
