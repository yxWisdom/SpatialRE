# import logging
# import os
# from collections import OrderedDict
# from itertools import islice, groupby, chain
#
# from utils.common_utils import eval_srl, to_srl_format
# from utils.seq_labeling_eval import classification_report
#
# import numpy as np
#
# logger = logging.getLogger(__name__)
#
#
# def _is_divider(line: str) -> bool:
#     return len(line.strip()) == 0
#
#
# def _get_padded_labels(tokens, label, use_x_tag=False):
#     if type(label) is str:
#         padding_label = label.replace('B-', 'I-')
#     else:
#         padding_label = label
#     return [label] + ["X" if use_x_tag and token.startswith("##") and type(label) is str else padding_label
#                       for token in tokens[1:]]
#
#
# class InputExample(object):
#     """A single training/test example for simple sequence classification."""
#
#     def __init__(self, guid, text, tokens=None, labels=None, features=None):
#         if features is None:
#             features = {}
#         self.guid = guid
#         self.text = text
#         self.tokens = tokens
#         self.labels = labels
#         self.features = features
#
#
# class InputFeatures(object):
#     """A single set of features of data."""
#
#     def __init__(self, input_ids, input_mask, segment_ids, label_ids, features, relative_positions=None,
#                  entity_mask=None):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label_ids = label_ids
#         self.features = features
#         self.relative_positions = relative_positions
#         self.entity_mask = entity_mask
#
#
# class EntityAwareSRLDataProcessor:
#     def __init__(self, data_dir, used_feature_names=None, use_x_tag=False, keep_span=True):
#         self.labels = set()
#         self.use_x_tag = use_x_tag
#         self.keep_span = keep_span
#
#         self.all_feature_names = ["trigger_mask", "tags", "rule_tags"]
#         self.used_feature_names = used_feature_names if used_feature_names is not None \
#             else self.all_feature_names
#
#         self.train_examples = self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")
#         self.dev_examples = self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")
#         self.test_examples = None
#
#         self._read_labels()
#
#     def get_train_examples(self):
#         return self.train_examples
#
#     def get_dev_examples(self):
#         return self.dev_examples
#
#     def get_test_examples(self):
#         return self.test_examples
#
#     def get_labels(self):
#         return self.labels
#
#     def get_features_size(self):
#         return [len(f_value) for f_value in self.feature_dict.values()]
#
#     def _read_labels(self):
#
#         def func(label):
#             a = label.split("-")
#             if len(a) == 1:
#                 return a
#             return ["B-" + a[-1], "I-" + a[-1]]
#
#         self.feature_dict = OrderedDict()
#         label_set = set(chain.from_iterable(map(lambda x: x.labels, self.train_examples)))
#         label_set = set(chain.from_iterable(map(func, label_set)))
#         if self.use_x_tag:
#             label_set.add("X")
#         self.labels = sorted(sorted(label_set, key=lambda x: x[:1]), key=lambda x: x[1:])
#
#         for key in self.used_feature_names:
#             feature_list = list(set(chain.from_iterable(map(lambda x: x.features[key], self.train_examples))))
#             if type(feature_list[0]) is str:
#                 feature_list = list(set(chain.from_iterable(map(func, feature_list))))
#                 if self.use_x_tag:
#                     feature_list.append("X")
#                 self.feature_dict[key] = sorted(sorted(feature_list, key=lambda x: x[:1]), key=lambda x: x[1:])
#             else:
#                 self.feature_dict[key] = sorted(feature_list)
#
#     @classmethod
#     def _read_data(cls, input_file):
#         with open(input_file, "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f if len(line.strip()) > 0]
#             return lines
#
#     def _create_example(self, lines, set_type):
#         examples = []
#         for i, line in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             columns = line.split('\t')
#
#             trigger_info = columns[0].split()
#             link_type = "LINK" if len(trigger_info) <= 2 else trigger_info[2].strip()
#             p_start, p_end = map(int, trigger_info[:2])
#             tokens = columns[1].split()
#             labels = columns[-1].split()
#             trigger_pos = ["O"] * len(tokens)
#             for j in range(p_start, p_end + 1):
#                 trigger_pos[j] = link_type
#
#             features = OrderedDict()
#             features["trigger_mask"] = trigger_pos
#             for (f_name, column) in zip(self.all_feature_names[1:], columns[2:-1]):
#                 if f_name in self.used_feature_names:
#                     features[f_name] = column.split()
#
#             assert len(tokens) == len(labels)
#             for k, v in features.items():
#                 assert len(tokens) == len(v)
#             text = " ".join(tokens)
#             examples.append(InputExample(guid=guid, text=text, tokens=tokens, labels=labels,
#                                          features=features))
#         return examples
#
#     def decode(self, predict_label_ids, tokenizer, set_type="dev"):
#         examples = self.dev_examples if set_type == 'dev' else self.test_examples
#         lines = []
#
#         for example, p_label_ids in zip(examples, predict_label_ids):
#             token_pieces_list = [tokenizer.tokenize(token) for token in example.tokens]
#             token_piece_lens = [len(piece) for piece in token_pieces_list]
#
#             it = iter(self.labels[label_id] for label_id in p_label_ids)
#             p_label_pieces_list = [list(islice(it, 0, i)) for i in token_piece_lens]
#
#             for token, tag, g_label, token_pieces, p_label_pieces in zip(example.tokens, example.features["tags"],
#                                                                          example.labels,
#                                                                          token_pieces_list, p_label_pieces_list):
#                 if self.keep_span:
#                     try:
#                         lines.append((token, tag, g_label, p_label_pieces[0]))
#                     except IndexError:
#                         lines.append((token, tag, g_label, "O"))
#                 else:
#                     tokens, p_labels = [], []
#                     for _token, _label in zip(token_pieces, p_label_pieces):
#                         if _token.startswith('##'):
#                             tokens[-1] += _token[2:]
#                         else:
#                             tokens.append(_token)
#                             p_labels.append(_label)
#                     g_labels = _get_padded_labels(token_pieces, g_label)
#                     lines.extend(list(zip(tokens, tag, g_labels, p_labels)))
#             lines.append([])
#         return lines
#
#     def evaluate(self, predict_label_ids, tokenizer, set_type="dev"):
#         eval_dict = {}
#         lines = self.decode(predict_label_ids, tokenizer, set_type)
#         _, _, golds, preds = zip(*(filter(lambda x: len(x) > 0, lines)))
#         eval_dict['srl_token'] = classification_report(y_true=list(golds), y_pred=list(preds), digits=4)
#         lines = list(map(lambda x: ' '.join(x), lines))
#
#         total_num, accuracy, total_correct_num, total_predict_num, total_gold_num, precision, recall, f_1 = eval_srl(lines)
#         eval_dict['srl_sentence'] = "total:{} accuracy:{:.4f}, correct:{},predict{},gold:{},precision:{:.4f}, recall{:.4f}, f1:{:.4f}". \
#             format(total_num, accuracy, total_correct_num, total_predict_num, total_gold_num,
#                    precision, recall, f_1)
#         eval_dict['accuracy'] = accuracy
#         eval_dict['f-1'] = f_1
#         return eval_dict
#
#     def save_predict(self, output_path, predict_label_ids, tokenizer, set_type="dev"):
#         lines = self.decode(predict_label_ids, tokenizer, set_type)
#         path = os.path.join(output_path, "predict.txt")
#         with open(path, 'w', encoding='utf-8') as writer:
#             for line in lines:
#                 writer.write(' '.join(line) + '\n')
#         to_srl_format(path)
#
#     @staticmethod
#     def find_entity_loc(mapping, value, max_seq_length):
#         try:
#             e_begin = mapping.index(value)
#             e_end = min(len(mapping) - 1 - mapping[::-1].index(value), max_seq_length)
#             return e_begin, e_end
#         except ValueError:
#             return -1, -1
#
#     @staticmethod
#     def convert_entity_row(mapping, loc, max_seq_length, max_distance=2):
#         lo, hi = loc
#         res = [max_distance] * max_seq_length
#         mas = [0] * max_seq_length
#         for i in range(max_seq_length):
#             if i < len(mapping):
#                 val = mapping[i]
#                 if val < lo - max_distance:
#                     res[i] = max_distance
#                 elif val < lo:
#                     res[i] = lo - val
#                 elif val <= hi:
#                     res[i] = 0
#                     mas[i] = 1
#                 elif val <= hi + max_distance:
#                     res[i] = val - hi + max_distance
#                 else:
#                     res[i] = 2 * max_distance
#             else:
#                 res[i] = 2 * max_distance
#         return res, mas
#
#     def prepare_extra_data(self, tags, token_to_ori_idx, max_distance, max_seq_length):
#         relative_positions = np.zeros([max_seq_length, max_seq_length], dtype=np.int8)
#         entity_mask = np.zeros([max_seq_length, max_seq_length], dtype=np.int8)
#
#         entities_loc = []
#
#         entity_start = 0
#
#         for i, tag in enumerate(tags):
#             if tag.startswith("B-"):
#                 entity_start = i
#             if tag != "O" and (i == len(tags) - 1 or tags[i + 1] == "O" or tags[i + 1].startswith("B-")):
#                 entity_end = i
#                 entities_loc.append((entity_start, entity_end))
#         for (e_start, e_end) in entities_loc:
#             relative_position, _ = self.convert_entity_row(token_to_ori_idx, (e_start, e_end),
#                                                            max_seq_length, max_distance)
#             sub_start, _ = self.find_entity_loc(token_to_ori_idx, e_start, max_seq_length)
#             _, sub_end = self.find_entity_loc(token_to_ori_idx, e_end, max_seq_length)
#
#             if sub_end < 0 or sub_end < 0:
#                 continue
#             relative_positions[:, sub_start: sub_end + 1] = np.expand_dims(relative_position, -1)
#             entity_mask[:, sub_start: sub_end + 1] = 1
#
#         for (e_start, e_end) in entities_loc:
#             relative_position, _ = self.convert_entity_row(token_to_ori_idx, (e_start, e_end),
#                                                            max_seq_length, max_distance)
#             sub_start, _ = self.find_entity_loc(token_to_ori_idx, e_start, max_seq_length)
#             _, sub_end = self.find_entity_loc(token_to_ori_idx, e_end, max_seq_length)
#
#             if sub_end < 0 or sub_end < 0:
#                 continue
#             relative_positions[sub_start: sub_end + 1, :] = relative_position
#             entity_mask[sub_start: sub_end + 1, :] = 1
#         return relative_positions, entity_mask
#
#     def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
#                                      cls_token_at_end=False,
#                                      cls_token='[CLS]',
#                                      cls_token_segment_id=1,
#                                      sep_token='[SEP]',
#                                      sep_token_extra=False,
#                                      pad_on_left=False,
#                                      pad_token=0,
#                                      pad_token_segment_id=0,
#                                      pad_token_label_id=-1,
#                                      sequence_a_segment_id=0,
#                                      sequence_b_segment_id=1,
#                                      mask_padding_with_zero=True,
#                                      ignore_cls_sep=True,
#                                      use_rel_pos=False,
#                                      max_rel_distance=2
#                                      ):
#         max_length = 0
#
#         label_map = {label: i for i, label in enumerate(self.labels)}
#         features_map = {}
#         feature_pad_label = {}
#         for k, v in self.feature_dict.items():
#             features_map[k] = {item: idx for idx, item in enumerate(v)}
#             if features_map[k].__contains__("O"):
#                 feature_pad_label[k] = features_map[k]["O"]
#             else:
#                 feature_pad_label[k] = features_map[k][0]
#
#         features = []
#         for (ex_index, example) in enumerate(examples):
#             if ex_index % 1000 == 0:
#                 logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#             tokens = []
#             labels = []
#
#             extra_raw_features = OrderedDict()
#             extra_features = OrderedDict()
#
#             for k in example.features.keys():
#                 extra_raw_features[k] = []
#
#             sub_tokens_list = []
#
#             for i, (token, label) in enumerate(zip(example.tokens, example.labels)):
#                 sub_tokens = tokenizer.tokenize(token)
#                 tokens.extend(sub_tokens)
#                 labels.extend(_get_padded_labels(sub_tokens, label, self.use_x_tag))
#                 for k, v in extra_raw_features.items():
#                     v.extend(_get_padded_labels(sub_tokens, example.features[k][i], self.use_x_tag))
#
#                 sub_tokens_list.append(sub_tokens)
#
#             if len(tokens) > max_length:
#                 max_length = len(tokens)
#
#             if len(tokens) > max_seq_length - 2:
#                 tokens = tokens[:(max_seq_length - 2)]
#
#             tokens = [cls_token] + tokens + [sep_token]
#
#             segment_ids = [sequence_a_segment_id] * len(tokens)
#             input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#             label_ids = [label_map[label] if label_map.__contains__(label) else -1 for label in labels]
#             label_ids = label_ids[:max_seq_length] + [pad_token_label_id] * (max_seq_length - len(label_ids))
#
#             for f_name, f_list in extra_raw_features.items():
#                 f_list = [features_map[f_name][item] for item in f_list]
#                 if ignore_cls_sep:
#                     f_list = f_list[:max_seq_length]
#                 else:
#                     if len(f_list) > max_seq_length - 2:
#                         f_list = f_list[:max_seq_length-2]
#                     f_list = [feature_pad_label[f_name]] + f_list + [feature_pad_label[f_name]]
#                 f_list = f_list + [feature_pad_label[f_name]] * (max_seq_length - len(f_list))
#                 extra_features[f_name] = f_list
#
#             input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#
#             padding_length = max_seq_length - len(input_ids)
#
#             input_ids = input_ids + ([pad_token] * padding_length)
#             input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
#             segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
#
#             # process extra relative positions information
#             relative_positions, entity_mask = None, None
#             if use_rel_pos:
#                 tags = example.features["tags"]
#                 if not ignore_cls_sep:
#                     sub_tokens_list = [[cls_token]] + sub_tokens_list + [[sep_token]]
#                     tags = ["O"] + tags + ["O"]
#
#                 sub_token_to_ori_idx = []
#                 for idx, sub_tokens in enumerate(sub_tokens_list):
#                     sub_token_to_ori_idx.extend([idx] * len(sub_tokens))
#
#                 relative_positions, entity_mask = self.prepare_extra_data(tags, sub_token_to_ori_idx,
#                                                                           max_distance=max_rel_distance,
#                                                                           max_seq_length=max_seq_length)
#
#             assert len(input_ids) == max_seq_length
#             assert len(input_mask) == max_seq_length
#             assert len(segment_ids) == max_seq_length
#             assert len(label_ids) == max_seq_length
#
#             for v in extra_features.values():
#                 assert len(v) == max_seq_length
#
#             np.set_printoptions(edgeitems=1000, linewidth=1000)
#             if ex_index < 5:
#                 logger.info("*** Example ***")
#                 logger.info("guid: %s" % example.guid)
#                 logger.info("tokens: %s" % " ".join(
#                     [str(x) for x in tokens]))
#                 logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#                 logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#                 logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#                 logger.info("labels: %s" % " ".join([str(x) for x in labels]))
#                 logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
#                 for k, v in extra_features.items():
#                     logger.info("%s: %s" % (k, " ".join([str(x) for x in extra_raw_features[k]])))
#                     logger.info("%s: %s" % (k, " ".join([str(x) for x in v])))
#
#                 # if use_rel_pos:
#                 #     logger.info("rel_pos:\n %s" % str(relative_positions))
#                 #     logger.info("entity mask\n%s" % str(entity_mask))
#
#             features.append(
#                 InputFeatures(input_ids=input_ids,
#                               input_mask=input_mask,
#                               segment_ids=segment_ids,
#                               label_ids=label_ids,
#                               features=extra_features,
#                               relative_positions=relative_positions,
#                               entity_mask=entity_mask))
#         print("max_length", max_length)
#         return features
#
#
# # def convert_examples_to_features(examples, label_list, tag_list, max_seq_length,
# #                                  tokenizer, use_x_tag=True,
# #                                  cls_token_at_end=False,
# #                                  cls_token='[CLS]',
# #                                  cls_token_segment_id=1,
# #                                  sep_token='[SEP]',
# #                                  sep_token_extra=False,
# #                                  pad_on_left=False,
# #                                  pad_token=0,
# #                                  pad_token_segment_id=0,
# #                                  pad_token_label_id=-1,
# #                                  sequence_a_segment_id=0,
# #                                  sequence_b_segment_id=1,
# #                                  mask_padding_with_zero=True):
# #     """ Loads a data file into a list of `InputBatch`s
# #         `cls_token_at_end` define the location of the CLS token:
# #             - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
# #             - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
# #         `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
# #     """
# #
# #     label_map = {label: i for i, label in enumerate(label_list)}
# #     # label_map.update({cls_token: label_map["O"], sep_token: label_map["O"]})
# #     tag_map = {tag: i for i, tag in enumerate(tag_list)}
# #
# #     features = []
# #     for (ex_index, example) in enumerate(examples):
# #         if ex_index % 1000 == 0:
# #             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
# #
# #         tokens, labels, tags, predicate_mask = [], [], [], [0] * max_seq_length
# #
# #         for token, label, tag in zip(example.tokens, example.labels, example.tags):
# #             sub_tokens = tokenizer.tokenize(token)
# #             tokens.extend(sub_tokens)
# #             labels.extend(_get_padded_labels(sub_tokens, label, use_x_tag))
# #             tags.extend(_get_padded_labels(sub_tokens, tag, use_x_tag))
# #
# #         # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
# #         special_tokens_count = 3 if sep_token_extra else 2
# #         if len(tokens) > max_seq_length - special_tokens_count:
# #             tokens = tokens[:(max_seq_length - special_tokens_count)]
# #
# #         tokens = tokens + [sep_token]
# #
# #         if sep_token_extra:
# #             # roberta uses an extra separator b/w pairs of sentences
# #             tokens += [sep_token]
# #
# #         segment_ids = [sequence_a_segment_id] * len(tokens)
# #
# #         if cls_token_at_end:
# #             tokens = tokens + [cls_token]
# #             segment_ids = segment_ids + [cls_token_segment_id]
# #         else:
# #             tokens = [cls_token] + tokens
# #             segment_ids = [cls_token_segment_id] + segment_ids
# #
# #         input_ids = tokenizer.convert_tokens_to_ids(tokens)
# #
# #         label_ids = [label_map[label] if label_map.__contains__(label) else -1 for label in labels]
# #         label_ids = label_ids[:max_seq_length] + [pad_token_label_id] * (max_seq_length - len(label_ids))
# #
# #         tag_ids = [tag_map[tag] for tag in tags]
# #         tag_ids = tag_ids[:max_seq_length] + [tag_map["O"]] * (max_seq_length - len(tag_ids))
# #
# #         predicate_mask = [1 if label.endswith("trigger") else 0 for label in labels]
# #         predicate_mask = predicate_mask[:max_seq_length] + [0] * (max_seq_length - len(predicate_mask))
# #
# #         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
# #
# #         padding_length = max_seq_length - len(input_ids)
# #
# #         if pad_on_left:
# #             input_ids = ([pad_token] * padding_length) + input_ids
# #             input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
# #             segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
# #         else:
# #             input_ids = input_ids + ([pad_token] * padding_length)
# #             input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
# #             segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
# #
# #         assert len(input_ids) == max_seq_length
# #         assert len(input_mask) == max_seq_length
# #         assert len(segment_ids) == max_seq_length
# #         assert len(label_ids) == max_seq_length
# #         assert len(predicate_mask) == max_seq_length
# #         assert len(tag_ids) == max_seq_length
# #
# #         if ex_index < 5:
# #             logger.info("*** Example ***")
# #             logger.info("guid: %s" % example.guid)
# #             logger.info("tokens: %s" % " ".join(
# #                 [str(x) for x in tokens]))
# #             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
# #             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
# #             logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
# #             logger.info("labels: %s" % " ".join([str(x) for x in labels]))
# #             logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
# #             logger.info("tag_ids: %s" % " ".join([str(x) for x in tag_ids]))
# #             logger.info("predict_mask: %s" % " ".join([str(x) for x in predicate_mask]))
# #
# #         features.append(
# #             InputFeatures(input_ids=input_ids,
# #                           input_mask=input_mask,
# #                           segment_ids=segment_ids,
# #                           label_ids=label_ids,
# #                           predicate_mask=predicate_mask,
# #                           tag_ids=tag_ids))
# #     return features
#
# # processors = {
# #     "entity_a"
# # }
