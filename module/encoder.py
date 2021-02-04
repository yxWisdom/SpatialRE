from typing import List, Tuple

import torch
from torch import nn

from pytorch_transformers import BertModel
from utils.bert_utils import bert_sequence_output_wrapper


class Word2VecEncoder(nn.Module):
    pass


class BertEncoder(nn.Module):
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
