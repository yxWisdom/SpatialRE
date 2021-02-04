# from __future__ import absolute_import, division, print_function
#
# import csv
# from itertools import islice, groupby
# import logging
# import os
# import sys
# from io import open
#
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
#
# logger = logging.getLogger(__name__)
#
#
# def _is_divider(line: str) -> bool:
#     if line.strip() == '':
#         return True
#     else:
#         first_token = line.split()[0]
#         if first_token == "-DOCSTART-":
#             return True
#         else:
#             return False
#
#
# def _get_padded_labels(tokens, label, use_x_tag=False):
#     padding_label = label.replace('B-', 'I-')
#     return [label] + ["X" if use_x_tag and token.startswith("##") else padding_label for token in tokens[1:]]
#
#
# class InputExample(object):
#     """A single training/test example for simple sequence classification."""
#
#     def __init__(self, guid, text, tokens=None, labels=None):
#         self.guid = guid
#         self.text = text
#         self.tokens = tokens
#         self.labels = labels
#
#
# class InputFeatures(object):
#     """A single set of features of data."""
#
#     def __init__(self, input_ids, input_mask, segment_ids, label_ids):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label_ids = label_ids
#
#
# class DataProcessor(object):
#     """Base class for data converters for sequence classification data sets."""
#
#     def get_train_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the train set."""
#         raise NotImplementedError()
#
#     def get_dev_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the dev set."""
#         raise NotImplementedError()
#
#     def get_labels(self):
#         """Gets the list of labels for this data set."""
#         raise NotImplementedError()
#
#     @classmethod
#     def _read_data(cls, input_file, delimiter=' '):
#         """Reads a tab separated value file."""
#         fields_list = []
#         with open(input_file, "r", encoding="utf-8") as f:
#             for is_divider, group_lines in groupby(f, _is_divider):
#                 if not is_divider:
#                     fields = [line.strip().split(delimiter) for line in group_lines]
#                     fields_list.append(fields)
#             return fields_list
#
#
# class BertNerDataProcessor(DataProcessor):
#     """Base class for data converters for sequence classification data sets."""
#
#     def __init__(self, data_dir, use_x_tag=True, keep_span=False):
#         self.labels = set()
#         self.use_x_tag = use_x_tag
#
#         self.train_examples = self._create_example(
#             self._read_data(os.path.join(data_dir, "train.txt")), "train"
#         )
#         self.dev_examples = self._create_example(
#             self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
#         )
#         self.test_examples = None
#
#         tmp_labels = set()
#         for example in self.train_examples:
#             for label in example.labels:
#                 tmp_labels.add(label)
#         if use_x_tag:
#             tmp_labels.add("X")
#         self.labels = sorted(tmp_labels)
#         self.keep_span = keep_span
#
#     def get_train_examples(self, data_dir):
#         return self.train_examples
#
#     def get_dev_examples(self, data_dir):
#         return self.dev_examples
#
#     def get_test_examples(self, data_dir):
#         return self.test_examples
#
#     def get_labels(self):
#         # return ['I-MISC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'X', 'I-LOC', 'B-MISC', 'B-LOC', 'O']
#         return self.labels
#
#     @staticmethod
#     def _create_example(fields_list, set_type):
#         examples = []
#         for i, fields in enumerate(fields_list):
#             fields = [list(field) for field in zip(*fields)]
#             tokens, labels = fields[0], fields[-1]
#             guid = "%s-%s" % (set_type, i)
#             text = " ".join(tokens)
#             examples.append(InputExample(guid=guid, text=text, tokens=tokens, labels=labels))
#         return examples
#
#     def save_predict(self, tokenizer, all_p_label_ids, out_path, set_type="dev"):
#         examples = self.dev_examples if set_type == 'dev' else self.test_examples
#         lines = []
#         for example, p_label_ids in zip(examples, all_p_label_ids):
#             # tokens = list(itertools.chain.from_iterable([tokenizer.tokenize(token) for token in example.tokens]))
#             token_pieces = [tokenizer.tokenize(token) for token in example.tokens]
#             token_pieces_len = [len(piece) for piece in token_pieces]
#
#             it = iter([self.labels[label_id] for label_id in p_label_ids[1:]])
#             p_label_pieces = [list(islice(it, 0, i)) for i in token_pieces_len]
#
#             for raw_token, raw_label, tokens, p_labels in zip(example.tokens, example.labels, token_pieces,
#                                                               p_label_pieces):
#                 if self.keep_span:
#                     try:
#                         lines.append((raw_token, raw_label, p_labels[0]))
#                     except:
#                         print()
#                         print(raw_token, raw_label, p_labels)
#                         print(list(zip(token_pieces, p_label_pieces)))
#                         print()
#                 else:
#                     tokens, _tmp_tokens = [], tokens
#                     p_labels, _tmp_p_labels = [], p_labels
#                     for token, label in zip(_tmp_tokens, _tmp_p_labels):
#                         if token.startswith("##"):
#                             tokens[-1] += token[2:]
#                         else:
#                             tokens.append(token)
#                             p_labels.append(label)
#                     g_labels = _get_padded_labels(tokens, raw_label)
#                     lines.extend(list(zip(tokens, g_labels, p_labels)))
#             lines.append([])
#         path = os.path.join(out_path, "predict.txt")
#         with open(path, 'w', encoding="utf-8") as writer:
#             for line in lines:
#                 writer.write(" ".join(line) + "\n")
#         return zip(*(filter(lambda x: len(x) > 0, lines)))
#
#
# def convert_examples_to_features(examples, label_list, max_seq_length,
#                                  tokenizer, use_x_tag=True,
#                                  cls_token_at_end=False,
#                                  cls_token='[CLS]',
#                                  cls_token_segment_id=1,
#                                  sep_token='[SEP]',
#                                  sep_token_extra=False,
#                                  pad_on_left=False,
#                                  pad_token=0,
#                                  pad_token_segment_id=0,
#                                  sequence_a_segment_id=0,
#                                  sequence_b_segment_id=1,
#                                  mask_padding_with_zero=True):
#     """ Loads a data file into a list of `InputBatch`s
#         `cls_token_at_end` define the location of the CLS token:
#             - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
#             - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
#         `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
#     """
#
#     label_map = {label: i for i, label in enumerate(label_list)}
#     label_map.update({cls_token: label_map["O"], sep_token: label_map["O"]})
#
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         tokens, labels = [], []
#
#         for token, label in zip(example.tokens, example.labels):
#             sub_tokens = tokenizer.tokenize(token)
#             tokens.extend(sub_tokens)
#             labels.extend(_get_padded_labels(sub_tokens, label, use_x_tag))
#
#         # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
#         special_tokens_count = 3 if sep_token_extra else 2
#         if len(tokens) > max_seq_length - special_tokens_count:
#             tokens = tokens[:(max_seq_length - special_tokens_count)]
#             labels = labels[:(max_seq_length - special_tokens_count)]
#
#         tokens = tokens + [sep_token]
#         labels = labels + [sep_token]
#
#         if sep_token_extra:
#             # roberta uses an extra separator b/w pairs of sentences
#             tokens += [sep_token]
#             labels += [sep_token]
#
#         segment_ids = [sequence_a_segment_id] * len(tokens)
#
#         if cls_token_at_end:
#             tokens = tokens + [cls_token]
#             labels = labels + [cls_token]
#             segment_ids = segment_ids + [cls_token_segment_id]
#         else:
#             tokens = [cls_token] + tokens
#             labels = [cls_token] + labels
#             segment_ids = [cls_token_segment_id] + segment_ids
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         label_ids = [label_map[label] if label_map.__contains__(label) else -1 for label in labels]
#
#         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#
#         padding_length = max_seq_length - len(input_ids)
#
#         if pad_on_left:
#             input_ids = ([pad_token] * padding_length) + input_ids
#             label_ids = ([-2] * padding_length) + label_ids
#             input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
#             segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
#         else:
#             input_ids = input_ids + ([pad_token] * padding_length)
#             label_ids = label_ids + ([-1] * padding_length)
#             input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
#             segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#         assert len(label_ids) == max_seq_length
#
#         if ex_index < 10:
#             logger.info("*** Example ***")
#             logger.info("guid: %s" % example.guid)
#             logger.info("tokens: %s" % " ".join(
#                 [str(x) for x in tokens]))
#             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#             logger.info("labels: %s" % " ".join([str(x) for x in labels]))
#             logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
#         features.append(
#             InputFeatures(input_ids=input_ids,
#                           input_mask=input_mask,
#                           segment_ids=segment_ids,
#                           label_ids=label_ids))
#     return features
#
#
# def simple_accuracy(preds, labels):
#     return accuracy_score(y_true=labels, y_pred=preds)
#
#
# def acc_and_f1(preds, labels, attention_labels=None):
#     acc = simple_accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds, labels=attention_labels, average='micro')
#     return {
#         "acc": acc,
#         "f1": f1,
#         "acc_and_f1": (acc + f1) / 2,
#     }
#
#
# processors = {
#     "ner_x_span": {
#         "name": BertNerDataProcessor,
#         "params": {
#             "use_x_tag": True,
#             "keep_span": True
#         },
#     },
#     "ner_x": {
#         "name": BertNerDataProcessor,
#         "params": {
#             "use_x_tag": True,
#             "keep_span": False
#         },
#     },
#     "ner_span": {
#         "name": BertNerDataProcessor,
#         "params": {
#             "use_x_tag": False,
#             "keep_span": True
#         },
#     },
#     "ner": {
#         "name": BertNerDataProcessor,
#         "params": {
#             "use_x_tag": False,
#             "keep_span": False
#         }
#     }
# }
