import numpy as np
import torch

# [CLS,t1, t2, ..., tn, SEP,...] -->  [t1, t2, ..., tn, SEP, ..., CLS]
# sequence_output shape: [batch_size * seq_length * embedding_dim]
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
        if i > 0 and pos == align_pos[i - 1] and label.startswith("B-"):
            new_labels.append("I" + label[1:])
        else:
            new_labels.append(label)
    return new_labels


def entity_relative_distance(token_map, span, max_seq_length, max_distance=2):
    low, high = span
    res = [max_distance] * max_seq_length
    for i in range(max_seq_length):
        if i < len(token_map):
            val = token_map[i]
            if val < low - max_distance:
                res[i] = max_distance
            elif val < low:
                res[i] = low - val
            elif val <= high:
                res[i] = 0
            elif val <= high + max_distance:
                res[i] = val - high + max_distance
            else:
                res[i] = 2 * max_distance
        else:
            res[i] = 2 * max_distance
    return res

# 注意与data processor中处理的区别
def entity_relative_distance_matrix(tags, token_to_ori_idx, max_distance, max_seq_length):
    entities_loc = []
    entity_start = 0

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            entity_start = i
        if tag != "O" and (i == len(tags) - 1 or tags[i + 1] == "O" or tags[i + 1].startswith("B-")):
            entity_end = i
            entities_loc.append((entity_start, entity_end))

    relative_positions = np.zeros([max_seq_length, max_seq_length], dtype=np.int8)
    entity_mask = np.zeros([max_seq_length, max_seq_length], dtype=np.int8)

    rel_pos_list = []
    for (e_start, e_end) in entities_loc:
        span = (token_to_ori_idx[e_start], token_to_ori_idx[e_end])
        relative_position = entity_relative_distance(token_to_ori_idx, span,
                                                        max_seq_length, max_distance)
        relative_positions[:, e_start: e_end + 1] = np.expand_dims(relative_position, -1)
        entity_mask[:, e_start: e_end + 1] = 1
        rel_pos_list.append(relative_position)

    for (e_start, e_end), relative_position in zip(entities_loc, rel_pos_list):
        relative_positions[e_start: e_end + 1, :] = relative_position
        entity_mask[e_start: e_end + 1, :] = 1

    return relative_positions, entity_mask

# input_tensor: [batch_size * seq_length]
# def softmax_decode(input_tensor, mask, pad_label_id):
#     input_tensor = input_tensor.detach()
#     output = torch.argmax(input_tensor, -1)
#     output[mask] = pad_label_id
#     indices = (output != pad_label_id).nonzero()
#     batch_size = input_tensor.size(0)
#     result = [[] for _ in range(batch_size)]
#     # result[1].append([1, 2])
#     for i in range(indices.size(0)):
#         batch, idx = indices[i].tolist()
#         label = int(output[batch, idx])
#         if label == pad_label_id:
#             continue
#         result[batch].append((idx, label))
#     return result
