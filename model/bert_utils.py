# [CLS,t1, t2, ..., tn, SEP,...] -->  [t1, t2, ..., tn, SEP, ..., CLS]
# sequence_output shape: [batch_size * seq_length * embedding_dim]
import torch


def bert_sequence_output_wrapper(sequence_output):
    return torch.cat((sequence_output[:, 1:, :], sequence_output[:, :1, :]), 1)


# [1,1,1,...,1,0,0,...] --> [1,...,1,0,0,...,0,0]
# attention_mask shape: [batch_size * seq_length]
def bert_mask_wrapper(attention_mask):
    return torch.cat((attention_mask[:, 2:], attention_mask[:, -2:]), -1)
