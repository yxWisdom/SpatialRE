# [CLS,t1, t2, ..., tn, SEP,...] -->  [t1, t2, ..., tn, SEP, ..., CLS]
# sequence_output shape: [batch_size * seq_length * embedding_dim]
import torch


def bert_sequence_output_wrapper(sequence_output):
    return torch.cat((sequence_output[:, 1:, :], sequence_output[:, :1, :]), 1)


# [1,1,1,...,1,0,0,...] --> [1,...,1,0,0,...,0,0]
# attention_mask shape: [batch_size * seq_length]
def bert_mask_wrapper(attention_mask):
    pad_tensor = torch.zeros(attention_mask.shape[0], 2, device=attention_mask.device).type_as(attention_mask)
    attention_mask = torch.cat((attention_mask[:, 2:], pad_tensor), -1)
    return attention_mask


def align_labels(align_pos, ori_labels):
    new_labels = []
    for i, pos in enumerate(align_pos):
        label = ori_labels[pos]
        if i > 0 and pos == align_pos[i-1] and label.startswith("B-"):
            new_labels.append("I"+label[1:])
        else:
            new_labels.append(label)
    return new_labels
