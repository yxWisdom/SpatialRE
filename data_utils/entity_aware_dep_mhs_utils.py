import bisect
import itertools
import logging
import os
from collections import OrderedDict
from collections import defaultdict
from itertools import groupby

import numpy as np
from dataclasses import dataclass

from utils.bert_utils import align_labels
from utils.spaceeval_utils import Metrics, Document

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    guid: str
    file: str
    text: str
    tokens: list
    element_ids: list
    features: dict
    triples: list
    dependency_tree: list


# class InputExample(object):
#     def __init__(self, guid, text, tokens=None, triples=None, features=None, element_ids=None, dependency_tree=None):
#         self.guid = guid
#         self.text = text
#         self.tokens = tokens
#         # self.labels = labels
#         self.features = features
#         self.triples = triples
#         self.element_ids =
#         self.file =
#         self.dependency_tree = dependency_tree


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, features, selection_matrix, relative_positions=None,
                 entity_mask=None, dependency_graph=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.label_ids = label_ids
        self.features = features
        self.relative_positions = relative_positions
        self.entity_mask = entity_mask
        self.selection_matrix = selection_matrix
        self.dependency_graph = dependency_graph


class EntityAwareMHSDataProcessor(object):
    def __init__(self, data_dir, tokenizer, used_feature_names=None, use_x_tag=False, keep_span=True, use_head=True):
        self.data_dir = data_dir
        self.use_head = use_head  # 该变量表示执行变量的头还是尾
        self.tokenizer = tokenizer

        self.relations = []
        self.dep_labels = []
        self.use_x_tag = use_x_tag
        self.keep_span = keep_span

        self.all_feature_names = ["tags", "trigger_mask", "rule_tags", ]
        self.used_feature_names = used_feature_names if used_feature_names is not None \
            else self.all_feature_names

        self._read_labels()

        self.train_examples = self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")
        self.dev_examples = self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")
        self.test_examples = None

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return self.relations

    def get_features_size(self):
        return [len(f_value) for f_value in self.feature_dict.values()]

    def _read_labels(self):

        def func(label):
            a = label.split("-")
            if len(a) == 1:
                return a
            return ["B-" + a[-1], "I-" + a[-1]]

        self.feature_dict = OrderedDict()

        examples = self._read_data(os.path.join(self.data_dir, "train.txt"))
        label_set = set()
        dep_label_set = {"O"}
        f_label_set_list = [set() for _ in self.used_feature_names]

        for example in examples:
            for tuples in example[1:]:
                relations = map(lambda x: x.strip(), tuples[-1][1:-1].split(","))
                label_set.update(relations)
                for f_idx, f_label in enumerate(tuples[2:2 + len(f_label_set_list)]):
                    f_label_set_list[f_idx].update(func(f_label))
                _, dep_label = tuples[-4].split(",")
                dep_label_set.add(dep_label)

        self.relations = list(sorted(label_set))
        self.dep_labels = list(sorted(dep_label_set))

        for i, f_name in enumerate(self.used_feature_names):
            f_label_set = f_label_set_list[i]
            self.feature_dict[f_name] = list(sorted(sorted(f_label_set, key=lambda x: x[:1]), key=lambda x: x[1:]))

        self.rel2IdMap = {rel: idx for idx, rel in enumerate(self.relations)}

    # def init(self):
    #     label_set = set()
    #     relation_set = set()
    #     examples = self._read_data(os.path.join(self.data_dir, "train.txt"))
    #     for example in examples:
    #         _, _, labels, relations_list, _ = (list(field) for field in zip(*example))
    #         for relations in relations_list:
    #             relations = relations[1:-1].replace(" ", "").split(",")
    #             relation_set.update(relations)
    #         label_set.update(labels)
    #
    #     self.labels = sorted(label_set)
    #     self.relations = sorted(relation_set)
    #     self.rel2IdMap = {relation: idx for idx, relation in enumerate(self.relations)}

    @classmethod
    def _read_data(cls, input_file, delimiter='\t'):
        raw_examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for is_divider, group_lines in groupby(f, lambda x: x.strip() == ''):
                if not is_divider:
                    raw_examples.append(
                        [list(map(lambda x: x.strip(), line.strip().split(delimiter))) for line in group_lines])
        return raw_examples

    # def trans_to_bert(self, tokens, features, triples):
    #     def head2new(old_head):
    #         if self.use_head:
    #             return bisect.bisect_left(align_token_pos, old_head)
    #             # return align_token_pos.index(old_head)
    #         else:
    #             # return len(align_token_pos) - list(reversed(align_token_pos)).index(old_head) - 1
    #             return len(align_token_pos) - bisect.bisect_right(align_token_pos, old_head)
    #
    #     align_token_pos, bert_tokens, align_features, new_triples = [], [], tuple([] for i in features), []
    #     token_pieces_list = [self.tokenizer.tokenize(token) for token in tokens]
    #     # align position
    #     for i, token_pieces in enumerate(token_pieces_list):
    #         align_token_pos.extend([i] * len(token_pieces))
    #         bert_tokens.extend(token_pieces)
    #
    #     for i, pos in enumerate(align_token_pos):
    #         for f_idx, feature in enumerate(features):
    #             f_prefix, f_label = feature[pos].partiton("-")
    #             if i > 0 and pos == align_token_pos[i - 1] and f_prefix == "B":
    #                 align_features[f_idx][i].append("I-" + f_label)
    #             else:
    #                 align_features[f_idx][i].append(feature[pos])
    #
    #     for triple in triples:
    #         new_triples.append((head2new(triple[0]), triple[1], head2new(triple[2])))
    #
    #     return bert_tokens, align_features, new_triples

    def decode_triples(self, tokens, triples):
        align_token_pos = []
        new_triples = []
        for idx, token in enumerate(tokens):
            sub_tokens = self.tokenizer.tokenize(token)
            align_token_pos.extend([idx] * len(sub_tokens))
        for (s, p, o) in triples:
            new_triples.append((align_token_pos[s], p, align_token_pos[o]))
        return new_triples

    def _create_example(self, raw_examples, set_type):
        examples = []
        for i, raw_example in enumerate(raw_examples):
            guid = "%s-%s" % (set_type, i)
            file = raw_example[0][0]
            tuples = tuple(list(field) for field in zip(*raw_example[1:]))
            tokens = tuples[1]
            element_ids, heads_list, relations_list = tuples[-3:]
            features = OrderedDict()

            for f_name, feature in zip(self.used_feature_names, tuples[2:]):
                features[f_name] = feature

            dependency_tree = []
            for dep_info in tuples[-4]:
                dep_tail, dep_label = dep_info.split(",")
                dependency_tree.append((int(dep_tail), dep_label))

            text = " ".join(tokens)
            triples = []
            for idx, (relations, heads) in enumerate(zip(relations_list, heads_list)):
                relations = relations[1:-1].replace(" ", "").split(",")
                heads = list(map(int, heads[1:-1].split(",")))
                for (relation, head) in zip(relations, heads):
                    if 'NA' not in relations:
                        triples.append((idx, self.rel2IdMap[relation], int(head)))

            # tokens, features, triples = self.trans_to_bert(tokens, features, triples)
            examples.append(
                InputExample(guid=guid, file=file, text=text, tokens=tokens, features=features, triples=triples,
                             element_ids=element_ids, dependency_tree=dependency_tree))
        return examples

    def save_predict_result(self, output_dir, pred_triples_list, include_gold=True, mark_error=True):
        lines = []
        for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
            # tokens, features, gold_triples = self.trans_from_bert(example.tokens, example.features, example.triples)
            # _, _, pred_triples = self.trans_from_bert(example.tokens, example.features, ori_pred_triples)
            tokens, features, gold_triples = example.tokens, example.features, example.triples
            pred_triples = self.decode_triples(example.tokens, ori_pred_triples)

            gold_relations, gold_heads = [[] for _ in tokens], [[] for _ in tokens]
            pred_relations, pred_heads = [[] for _ in tokens], [[] for _ in tokens]

            for (s, p, o) in gold_triples:
                gold_relations[s].append(self.relations[p])
                gold_heads[s].append(o)
            for (s, p, o) in pred_triples:
                pred_relations[s].append(self.relations[p])
                pred_heads[s].append(o)

            for i, (token, label, gold_relation, gold_head, pred_relation, pred_head) in \
                    enumerate(zip(tokens, features["tags"], gold_relations, gold_heads, pred_relations, pred_heads)):
                gold_relation = ["NA"] if len(gold_relation) == 0 else gold_relation
                gold_head = [i] if len(gold_head) == 0 else gold_head
                pred_relation = ["NA"] if len(pred_relation) == 0 else pred_relation
                pred_head = [i] if len(pred_head) == 0 else pred_head

                if len(gold_relation) > 1:
                    pair = sorted(zip(gold_relation, gold_head))
                    gold_relation = list(map(lambda x: x[0], pair))
                    gold_head = list(map(lambda x: x[1], pair))

                if len(pred_relation) > 1:
                    pair = sorted(zip(pred_relation, pred_head))
                    pred_relation = list(map(lambda x: x[0], pair))
                    pred_head = list(map(lambda x: x[1], pair))

                line = f'{i}\t{token.ljust(15)}\t{label.ljust(16)}\t'
                # line = "{i}\t{token}\t{label}\t".format(token=token.ljust(15), label=label.ljust(16))
                if include_gold:
                    gold_relation_str = "[{}]".format(",".join(gold_relation))
                    line += "{}\t{}\t".format(gold_relation_str.ljust(20), str(list(gold_head)).ljust(10))
                pred_relation_str = "[{}]".format(",".join(pred_relation))
                line += "{}\t{}\t".format(pred_relation_str.ljust(20), str(list(pred_head)).ljust(10))

                if mark_error and (pred_head != gold_head or pred_relation != gold_relation):
                    line += "ERROR"
                lines.append(line)
            lines.append("")
        output_path = os.path.join(output_dir, "predict.txt")
        with open(output_path, "w", encoding="utf-8") as file:
            for line in lines:
                file.write(line + "\n")

    # def get_links(self, spo_triples):
    #     mover_type = "mover"
    #     trajector_type = "trajector"
    #     landmark_type = "landmark"
    #     trigger_type = "trigger"
    #     located_in_type = "locatedIn"
    #
    #     move_links = {}
    #     non_move_links = {}
    #
    #     move_link_triples = set()
    #     non_move_link_tuples = set()
    #     no_trigger_triples = set()
    #
    #     for (s, p, o) in spo_triples:
    #         if p in [self.rel2IdMap[trajector_type], self.rel2IdMap[landmark_type]]:
    #             if not non_move_links.__contains__(o):
    #                 non_move_links[o] = {trajector_type: [], "trigger": o, landmark_type: []}
    #             non_move_links[o][self.relations[p]].append(s)
    #         elif p == self.rel2IdMap[mover_type]:
    #             if not move_links.__contains__(o):
    #                 move_links[o] = {mover_type: [], trigger_type: o}
    #             move_links[o][self.relations[p]].append(s)
    #         elif p == self.rel2IdMap[located_in_type]:
    #             no_trigger_triples.add(("NoTriggerLink", s, o))
    #
    #         # 处理逆关系
    #         # elif p == self.rel2IdMap[located_in_type]
    #
    #     for trigger, link in move_links.items():
    #         movers = link[mover_type] if len(link[mover_type]) else [None]
    #         for mover in movers:
    #             move_link_triples.add(("MoveLink", mover, trigger))
    #     for trigger, link in non_move_links.items():
    #         trajectors = link[trajector_type] if len(link[trajector_type]) > 0 else [None]
    #         landmarks = link[landmark_type] if len(link[landmark_type]) > 0 else [None]
    #         for trajector in trajectors:
    #             for landmark in landmarks:
    #                 non_move_link_tuples.add(("NonMoveLink", trajector, trigger, landmark))
    #
    #     return move_link_triples, non_move_link_tuples, no_trigger_triples
    #
    # def evaluation(self, pred_triples_list):
    #     def calculate_prf(gold, pred, correct):
    #         precision = correct / pred if pred != 0 else 0.0
    #         recall = correct / gold if gold != 0 else 0.0
    #         f_1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0.0
    #         return precision, recall, f_1
    #
    #     link_types = ["MoveLink", "NonMoveLink", "NoTriggerLink", "All"]
    #     eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}
    #
    #     for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
    #         gold_triples = example.triples
    #         pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
    #         gold_links_tuples = self.get_links(gold_triples)
    #         pred_links_tuples = self.get_links(pred_triples)
    #
    #         for (link_type, gold_links, pred_links) in zip(link_types[:-1], gold_links_tuples, pred_links_tuples):
    #             eval_dict[link_type]["predict"] += len(pred_links)
    #             eval_dict[link_type]["gold"] += len(gold_links)
    #             eval_dict[link_type]["correct"] += len(gold_links & pred_links)
    #
    #             eval_dict["All"]["predict"] += len(pred_links)
    #             eval_dict["All"]["gold"] += len(gold_links)
    #             eval_dict["All"]["correct"] += len(gold_links & pred_links)
    #
    #     for k, v in eval_dict.items():
    #         p, r, f = calculate_prf(v["gold"], v["predict"], v["correct"])
    #         eval_dict[k]["p"] = p
    #         eval_dict[k]["r"] = r
    #         eval_dict[k]["f1"] = f
    #     return eval_dict

    @staticmethod
    def find_entity_loc(mapping, value, max_seq_length):
        try:
            e_begin = mapping.index(value)
            e_end = min(len(mapping) - 1 - mapping[::-1].index(value), max_seq_length)
            return e_begin, e_end
        except ValueError:
            return -1, -1

    @staticmethod
    def convert_entity_row(mapping, loc, max_seq_length, max_distance=2):
        lo, hi = loc
        res = [max_distance] * max_seq_length
        mas = [0] * max_seq_length
        for i in range(max_seq_length):
            if i < len(mapping):
                val = mapping[i]
                if val < lo - max_distance:
                    res[i] = max_distance
                elif val < lo:
                    res[i] = lo - val
                elif val <= hi:
                    res[i] = 0
                    mas[i] = 1
                elif val <= hi + max_distance:
                    res[i] = val - hi + max_distance
                else:
                    res[i] = 2 * max_distance
            else:
                res[i] = 2 * max_distance
        return res, mas

    def prepare_extra_data(self, tags, token_to_ori_idx, max_distance, max_seq_length):
        relative_positions = np.zeros([max_seq_length, max_seq_length], dtype=np.int8)
        entity_mask = np.zeros([max_seq_length, max_seq_length], dtype=np.int8)

        entities_loc = []

        entity_start = 0

        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                entity_start = i
            if tag != "O" and (i == len(tags) - 1 or tags[i + 1] == "O" or tags[i + 1].startswith("B-")):
                entity_end = i
                entities_loc.append((entity_start, entity_end))
        for (e_start, e_end) in entities_loc:
            relative_position, _ = self.convert_entity_row(token_to_ori_idx, (e_start, e_end),
                                                           max_seq_length, max_distance)
            sub_start, _ = self.find_entity_loc(token_to_ori_idx, e_start, max_seq_length)
            _, sub_end = self.find_entity_loc(token_to_ori_idx, e_end, max_seq_length)

            if sub_end < 0 or sub_end < 0:
                continue
            relative_positions[:, sub_start: sub_end + 1] = np.expand_dims(relative_position, -1)
            entity_mask[:, sub_start: sub_end + 1] = 1

        for (e_start, e_end) in entities_loc:
            relative_position, _ = self.convert_entity_row(token_to_ori_idx, (e_start, e_end),
                                                           max_seq_length, max_distance)
            sub_start, _ = self.find_entity_loc(token_to_ori_idx, e_start, max_seq_length)
            _, sub_end = self.find_entity_loc(token_to_ori_idx, e_end, max_seq_length)

            if sub_end < 0 or sub_end < 0:
                continue
            relative_positions[sub_start: sub_end + 1, :] = relative_position
            entity_mask[sub_start: sub_end + 1, :] = 1
        return relative_positions, entity_mask

    def align_selection_triple(self, align_token_pos, triples):
        def head2new(old_head):
            if self.use_head:
                return align_token_pos.index(old_head)
            else:
                return len(align_token_pos) - list(reversed(align_token_pos)).index(old_head) - 1

        return [(head2new(triple[0]), triple[1], head2new(triple[2])) for triple in triples]

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sep_token_extra=False,
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     pad_token_label_id=-1,
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     mask_padding_with_zero=True,
                                     ignore_cls_sep=True,
                                     use_rel_pos=False,
                                     max_rel_distance=2
                                     ):
        max_length = 0

        rel2idMap = {label: i for i, label in enumerate(self.relations)}
        dep2id_map = {label: i for i, label in enumerate(self.dep_labels)}

        features_map = {}
        feature_pad_label = {}
        for k, v in self.feature_dict.items():
            features_map[k] = {item: idx for idx, item in enumerate(v)}
            if features_map[k].__contains__("O"):
                feature_pad_label[k] = features_map[k]["O"]
            else:
                feature_pad_label[k] = features_map[k][0]

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 1000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = []

            extra_raw_features = OrderedDict()
            extra_features = OrderedDict()

            align_token_pos = []
            sub_tokens_list = []

            for i, token in enumerate(example.tokens):
                sub_tokens = tokenizer.tokenize(token)
                tokens.extend(sub_tokens)
                sub_tokens_list.append(sub_tokens)
                align_token_pos.extend([i] * len(sub_tokens))

            for f_name, f_list in example.features.items():
                extra_raw_features[f_name] = align_labels(align_token_pos, f_list)

            if len(tokens) > max_length:
                max_length = len(tokens)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]

            tokens = [cls_token] + tokens + [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # label_ids = [label_map[label] for label in labels]
            # label_ids = label_ids + [label_map["O"]] * (max_seq_length - len(label_ids))

            for f_name, f_list in extra_raw_features.items():
                f_list = [features_map[f_name][item] for item in f_list]
                f_pad_label = feature_pad_label[f_name]
                if ignore_cls_sep:
                    f_list = f_list[:max_seq_length]
                else:
                    f_list = [f_pad_label] + f_list[:max_seq_length - 2] + [f_pad_label]
                f_list = f_list + [f_pad_label] * (max_seq_length - len(f_list))
                extra_features[f_name] = f_list

            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)

            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            selection_matrix = np.zeros((max_seq_length, len(self.relations), max_seq_length), dtype=np.int)
            none = rel2idMap["NA"]
            selection_matrix[:, none, :] = 1
            selection_triples = self.align_selection_triple(align_token_pos, example.triples)
            for (s, p, o) in selection_triples:
                if s > max_seq_length - 1 or o > max_seq_length - 1:
                    continue
                selection_matrix[s, p, o] = 1
                selection_matrix[s, none, o] = 0

            dependency_graph = np.zeros((max_seq_length, max_seq_length), dtype=np.int)
            for dep_head, (dep_tail, dep_label) in enumerate(example.dependency_tree):
                new_dep_head = bisect.bisect_left(align_token_pos, dep_head)
                new_dep_tail = bisect.bisect_left(align_token_pos, dep_tail)
                dep_label_id = dep2id_map.get(dep_label)
                if new_dep_head >= max_seq_length or new_dep_tail >= max_seq_length:
                    continue
                dependency_graph[new_dep_tail, new_dep_head] = dep_label_id

            relative_positions, entity_mask = None, None
            if use_rel_pos:
                tags = example.features["tags"]
                if not ignore_cls_sep:
                    sub_tokens_list = [[cls_token]] + sub_tokens_list + [[sep_token]]
                    tags = ["O"] + tags + ["O"]

                sub_token_to_ori_idx = []
                for idx, sub_tokens in enumerate(sub_tokens_list):
                    sub_token_to_ori_idx.extend([idx] * len(sub_tokens))

                relative_positions, entity_mask = self.prepare_extra_data(tags, sub_token_to_ori_idx,
                                                                          max_distance=max_rel_distance,
                                                                          max_seq_length=max_seq_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            for v in extra_features.values():
                assert len(v) == max_seq_length

            np.set_printoptions(edgeitems=1000, linewidth=1000)
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                # logger.info("labels: %s" % " ".join([str(x) for x in labels]))
                # logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                logger.info("triples: %s" % " ".join([str(x) for x in example.triples]))
                for k, v in extra_features.items():
                    logger.info("%s: %s" % (k, " ".join([str(x) for x in extra_raw_features[k]])))
                    logger.info("%s: %s" % (k, " ".join([str(x) for x in v])))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              features=extra_features,
                              relative_positions=relative_positions,
                              entity_mask=entity_mask,
                              selection_matrix=selection_matrix,
                              dependency_graph=dependency_graph))
        print("max_length", max_length)
        return features


class EntityAwareMHSDataProcessor_2(EntityAwareMHSDataProcessor):
    def __init__(self, data_dir, tokenizer, used_feature_names=None, use_x_tag=False, keep_span=True, use_head=True):
        super().__init__(data_dir, tokenizer, used_feature_names, use_x_tag, keep_span, use_head)
        self.role2link_dict = {"trajector_Q": "QSLINK", "landmark_Q": "QSLINK",
                               "trajector_O": "OLINK", "landmark_O": "OLINK",
                               "QSLINK": "NoTrigger", "OLINK": "NoTrigger",
                               "mover": "MOVELINK", "source": "MOVELINK", "midPoint": "MOVELINK", "goal": "MOVELINK",
                               "motion_signalID": "MOVELINK", "pathID": "MOVELINK", "ground": "MOVELINK"}
        self.link2role_dict = {
            "QSLINK": ["trajector_Q", "landmark_Q"],
            "OLINK": ["trajector_O", "landmark_O"],
            "MOVELINK": ["mover", "source", "goal", "ground", "midPoint", "pathID", "motion_signalID"],
            # "NoTrigger": ["locatedIn"],
        }

    def get_links(self, spo_triples):
        raw_link_dict = defaultdict(dict)
        link2tuples = defaultdict(set)
        for (s, p, o) in spo_triples:
            role_type = self.relations[p]
            if role_type not in self.role2link_dict:
                continue
            link_type = self.role2link_dict[role_type]
            if link_type == "NoTrigger":
                link2tuples[link_type].add((link_type, '', s, o))
                # comment: role_type is QSLINK or OLINK
                link2tuples[role_type].add((role_type, '', s, o))
            else:
                if s not in raw_link_dict[link_type]:
                    raw_link_dict[link_type][s] = OrderedDict([(role, set()) for role in self.link2role_dict[link_type]])
                raw_link_dict[link_type][s][role_type].add(o)

        for link_type, link_dict in raw_link_dict.items():
            for trigger, role_dict in link_dict.items():
                role_group = ()
                for role_type, role_set in role_dict.items():
                    # if role_type in ["pathID", "midPoint", "motion_signalID"]:
                    if role_type == "motion_signalID":
                        role_tuples = ",".join(sorted(map(str, role_set)))
                        role_group += ([role_tuples],)
                    else:
                        if not role_set:
                            role_set.add("")
                        role_group += (role_set,)
                for tuple_ in itertools.product(*role_group):
                    link2tuples[link_type].add((link_type, trigger,) + tuple_)
                    # # TODO: 注意NoTrigger
                    # if link_type == "NoTrigger":
                    #     trajector = trigger
                    #     link2tuples[link_type].add((link_type, '', trajector) + tuple_)
                    #     link2tuples["QSLINK"].add(("QSLINK", '', trajector) + tuple_)
                    # else:


        return link2tuples

    def post_process(self, triples, tags, element_ids=None):
        link2tuples = defaultdict(set)
        trigger_set = set()
        for (s, p, o) in triples:
            role_type = self.relations[p]
            if role_type in self.role2link_dict:
                trigger_set.add(s)

        for i, tag in enumerate(tags):
            trigger = i if element_ids is None else element_ids[i]
            if tag == "B-MOTION" and trigger not in trigger_set:
                link2tuples["MOVELINK"].add(("MOVELINK", trigger, "", "", "", "", "", "", ""))

        return link2tuples

    @staticmethod
    def calculate_prf(gold, pred, correct):
        precision = correct / pred if pred != 0 else 0.0
        recall = correct / gold if gold != 0 else 0.0
        f_1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0.0
        return precision, recall, f_1

    def evaluate(self, pred_triples_list, allow_null_mover=True):
        link_types = ["MOVELINK", "QSLINK", "OLINK", "NoTrigger", "OVERALL"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}
        # TODO:
        # eval_optional_roles = "source" in self.relations
        for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
            gold_triples = example.triples
            gold_link_dict = self.get_links(gold_triples)
            other_gold_link_dict = self.post_process(gold_triples, example.features["tags"])

            pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
            pred_link_dict = self.get_links(pred_triples)
            other_pred_link_dict = self.post_process(pred_triples, example.features["tags"]
                                                     ) if allow_null_mover else defaultdict(set)

            for link_type in link_types:
                gold_links = gold_link_dict[link_type] | other_gold_link_dict[link_type]
                pred_links = pred_link_dict[link_type] | other_pred_link_dict[link_type]

                eval_dict[link_type]["predict"] += len(pred_links)
                eval_dict[link_type]["gold"] += len(gold_links)
                eval_dict[link_type]["correct"] += len(gold_links & pred_links)

                if link_type != "NoTrigger":
                    eval_dict["OVERALL"]["predict"] += len(pred_links)
                    eval_dict["OVERALL"]["gold"] += len(gold_links)
                    eval_dict["OVERALL"]["correct"] += len(gold_links & pred_links)

        # eval_dict["QSLINK"] = {k: eval_dict["QSLINK"][k] + eval_dict["NoTrigger"][k] for k in eval_dict["QSLINK"]}

        for k, v in eval_dict.items():
            p, r, f = self.calculate_prf(v["gold"], v["predict"], v["correct"])
            eval_dict[k]["p"] = p
            eval_dict[k]["r"] = r
            eval_dict[k]["f1"] = f
        return eval_dict

    def evaluate_exact(self, pred_triples_list, metric: Metrics, eval_optional_roles=True, eval_link_attr=False):
        link_types = ["MOVELINK", "QSLINK", "OLINK", "NoTrigger", "OVERALL"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}
        file2links = defaultdict(lambda: defaultdict(set))

        if eval_optional_roles and "source" not in self.relations:
            eval_optional_roles = False

        eval_null_mover = metric != Metrics.STRICT

        for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
            file = example.file
            tags = example.features["tags"]
            element_ids = example.element_ids

            pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
            pred_triples = [(element_ids[s], p, element_ids[o]) for s, p, o in pred_triples]
            pred_link_dict = self.get_links(pred_triples)
            other_pred_link_dict = self.post_process(pred_triples, tags, element_ids
                                                     ) if eval_null_mover else defaultdict(set)

            # pred_triples = example.triples
            # pred_triples = [(element_ids[s], p, element_ids[o]) for s, p, o in pred_triples]
            # pred_link_dict = self.get_links(pred_triples)
            # other_pred_link_dict = self.post_process(pred_triples, tags, element_ids,
            #                                          ) if eval_null_mover else defaultdict(set)

            for link_type in link_types:
                pred_links = pred_link_dict[link_type] | other_pred_link_dict[link_type]

                if link_type == "MOVELINK" and (not eval_optional_roles or metric == Metrics.OFFICIAL):
                    pred_links = set(map(lambda x: x[:3], pred_links))

                if metric == Metrics.STRICT:
                    if link_type == "MOVELINK":
                        pred_links = set(filter(lambda x: x[2] != "", pred_links))
                    else:
                        pred_links = set(filter(lambda x: x[2] != "" and x[3] != "", pred_links))

                if metric == Metrics.OFFICIAL:
                    if link_type != "MOVELINK":
                        pred_links = set(filter(lambda x: x[1] != "" and x[2] != "" and x[3] != "", pred_links))

                file2links[file][link_type].update(pred_links)

        for file, pred_link_dict in file2links.items():
            path = os.path.join(self.data_dir, 'xml', file)
            document = Document(path, metric, eval_optional_roles, eval_null_mover, eval_link_attr)
            gold_link_dict = document.load_links_from_file()
            # gold_link_dict = load_links_from_file(path, metric, eval_optional_roles, eval_null_mover)

            # pred_link_dict["QSLINK"].extend(pred_link_dict["NoTrigger"])
            for link_type in link_types:
                gold_links = gold_link_dict[link_type]
                pred_links = pred_link_dict[link_type]

                correct_links = gold_links & pred_links
                gold_num, pred_num, correct_num = len(gold_links), len(pred_links), len(correct_links)

                # if pred_num != gold_num and link_type != "MOVELINK":
                #     print(1)
                # if pred_num != correct_num:
                #     print(file)
                #     print("gold_link")
                #     gold_link_counter = Counter(gold_links) - Counter(pred_links)
                #     for link in gold_link_counter.elements():
                #         print(link)
                #     print("pred_link")
                #     pred_link_counter = Counter(pred_links) - Counter(gold_links)
                #     for link in pred_link_counter.elements():
                #         print(link)
                #     print()

                eval_dict[link_type]["predict"] += pred_num
                eval_dict[link_type]["gold"] += gold_num
                eval_dict[link_type]["correct"] += correct_num

                if link_type != "NoTrigger":
                    eval_dict["OVERALL"]["predict"] += pred_num
                    eval_dict["OVERALL"]["gold"] += gold_num
                    eval_dict["OVERALL"]["correct"] += correct_num

        # eval_dict["QSLINK"] = {k: eval_dict["QSLINK"][k] + eval_dict["NoTrigger"][k] for k in eval_dict["QSLINK"]}

        for k, v in eval_dict.items():
            p, r, f = self.calculate_prf(v["gold"], v["predict"], v["correct"])
            eval_dict[k]["p"] = p
            eval_dict[k]["r"] = r
            eval_dict[k]["f1"] = f
        return eval_dict

    # def evaluate_exact(self, pred_triples_list, allow_null_mover=False, eval_null_roles=True, eval_optional_roles=True):
    #
    #     link_types = ["MOVELINK", "QSLINK", "OLINK", "NoTrigger", "OVERALL"]
    #     eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}
    #     file2links = defaultdict(lambda: defaultdict(list))
    #
    #     if eval_optional_roles and "source" not in self.relations:
    #         return {}
    #
    #     # eval_optional_roles = "source" in self.relations
    #     for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
    #         file = example.file
    #         tags = example.features["tags"]
    #         element_ids = example.element_ids
    #
    #         pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
    #         pred_triples = [(element_ids[s], p, element_ids[o]) for s, p, o in pred_triples]
    #         pred_link_dict = self.get_links(pred_triples, eval_optional_roles)
    #         other_pred_link_dict = self.post_process(pred_triples, tags,
    #                                                  element_ids) if allow_null_mover else defaultdict(set)
    #
    #         # pred_triples = example.triples
    #         # pred_triples = [(element_ids[s], p, element_ids[o]) for s, p, o in pred_triples]
    #         # pred_link_dict = self.get_links(pred_triples, eval_optional_roles)
    #         # other_pred_link_dict = self.post_process(pred_triples, tags,
    #         #                                          element_ids) if allow_null_mover else defaultdict(set)
    #
    #         for link_type in link_types:
    #             pred_links = pred_link_dict[link_type] | other_pred_link_dict[link_type]
    #             file2links[file][link_type].extend(pred_links)
    #
    #     for file, pred_link_dict in file2links.items():
    #         path = os.path.join(self.data_dir, 'xml', file)
    #         gold_link_dict = load_gold_triple(path, eval_optional_roles, eval_null_roles)
    #
    #         # pred_link_dict["QSLINK"].extend(pred_link_dict["NoTrigger"])
    #
    #         for link_type in link_types:
    #             gold_links = gold_link_dict[link_type]
    #             pred_links = pred_link_dict[link_type]
    #
    #             correct_links = Counter(gold_links) & Counter(pred_links)
    #             gold_num, pred_num, correct_num = len(gold_links), len(pred_links), sum(correct_links.values())
    #
    #             # if link_type == "MOVELINK" and pred_num != correct_num:
    #             #     print(file)
    #             #     print("gold_link")
    #             #     gold_link_counter = Counter(gold_links) - Counter(pred_links)
    #             #     for link in gold_link_counter.elements():
    #             #         print(link)
    #             #
    #             #     print("pred_link")
    #             #     pred_link_counter = Counter(pred_links) - Counter(gold_links)
    #             #     for link in pred_link_counter.elements():
    #             #         print(link)
    #             #
    #             #     print()
    #
    #             eval_dict[link_type]["predict"] += pred_num
    #             eval_dict[link_type]["gold"] += gold_num
    #             eval_dict[link_type]["correct"] += correct_num
    #
    #             if link_type != "NoTrigger":
    #                 eval_dict["OVERALL"]["predict"] += pred_num
    #                 eval_dict["OVERALL"]["gold"] += gold_num
    #                 eval_dict["OVERALL"]["correct"] += correct_num
    #
    #     # eval_dict["QSLINK"] = {k: eval_dict["QSLINK"][k] + eval_dict["NoTrigger"][k] for k in eval_dict["QSLINK"]}
    #
    #     for k, v in eval_dict.items():
    #         p, r, f = self.calculate_prf(v["gold"], v["predict"], v["correct"])
    #         eval_dict[k]["p"] = p
    #         eval_dict[k]["r"] = r
    #         eval_dict[k]["f1"] = f
    #     return eval_dict

    # def get_links(self, spo_triples, decompose=True):
    #     move_links = {}
    #     t_links = {}
    #     o_links = {}
    #
    #     move_link_triples = set()
    #     t_link_triples = set()
    #     o_link_triples = set()
    #     no_trigger_triples = set()
    #
    #     for (s, p, o) in spo_triples:
    #         rel_type = self.relations[p]
    #         if not self.role2link_dict.__contains__(rel_type):
    #             continue
    #         # if rel_type.startswith("trigger"):
    #         #     continue
    #         link_type = self.role2link_dict[rel_type]
    #
    #         if link_type == "QSLINK":
    #             if not t_links.__contains__(s):
    #                 t_links[s] = {role: set() for role in self.link2role_dict[link_type]}
    #             t_links[s][rel_type].add(o)
    #         elif link_type == "OLINK":
    #             if not o_links.__contains__(s):
    #                 o_links[s] = {role: set() for role in self.link2role_dict[link_type]}
    #             o_links[s][rel_type].add(o)
    #         elif link_type == "MOVELINK":
    #             if not move_links.__contains__(s):
    #                 move_links[s] = {role: set() for role in self.link2role_dict[link_type]}
    #             move_links[s][rel_type].add(o)
    #         elif link_type == "NoTrigger":
    #             no_trigger_triples.add(("NoTriggerLink", s, o))
    #
    #     if not decompose:
    #         return move_links, t_links, o_links, no_trigger_triples
    #
    #     for trigger, link in move_links.items():
    #         movers = link["mover"] if len(link["mover"]) > 0 else [None]
    #         goals = link["goal"] if len(link["goal"]) > 0 else [None]
    #         mid_points = [str(link["midPoint"])] if len(link["midPoint"]) > 0 else [None]
    #         sources = link["source"] if len(link["source"]) > 0 else [None]
    #         paths = [str(link["pathID"])] if len(link["pathID"]) > 0 else [None]
    #         motion_signals = [str(link["motion_signalID"])] if len(link["motion_signalID"]) > 0 else [None]
    #         grounds = link["ground"] if len(link["ground"]) > 0 else [None]
    #
    #         for t in itertools.product(movers, sources, mid_points, goals, paths, motion_signals, grounds):
    #             move_link_triples.add(("MoveLink", trigger,) + t)
    #
    #     for trigger, link in t_links.items():
    #         trajectors = link["trajector_Q"] if len(link["trajector_Q"]) > 0 else [None]
    #         landmarks = link["landmark_Q"] if len(link["landmark_Q"]) > 0 else [None]
    #         for trajector in trajectors:
    #             for landmark in landmarks:
    #                 t_link_triples.add(("QSLINK", trajector, trigger, landmark))
    #
    #     for trigger, link in o_links.items():
    #         trajectors = link["trajector_O"] if len(link["trajector_O"]) > 0 else [None]
    #         landmarks = link["landmark_O"] if len(link["landmark_O"]) > 0 else [None]
    #         for trajector in trajectors:
    #             for landmark in landmarks:
    #                 o_link_triples.add(("OLINK", trajector, trigger, landmark))
    #
    #     return move_link_triples, t_link_triples, o_link_triples, no_trigger_triples
    #
    # def post_process(self, triples, tags, decompose=True):
    #     trigger_dict = collections.defaultdict(set)
    #     for (s, p, o) in triples:
    #         rel_type = self.relations[p]
    #         if self.role2link_dict.__contains__(rel_type):
    #             link_type = self.role2link_dict[rel_type]
    #             trigger_dict[s].add(link_type)
    #
    #     move_links_dict = {}
    #     t_links_dict = {}
    #     o_links_dict = {}
    #
    #     move_link_triples = set()
    #     t_link_triples = set()
    #     o_link_triples = set()
    #     no_trigger_triples = set()
    #
    #     for i, tag in enumerate(tags):
    #         if tag == "B-MOTION":
    #             if not trigger_dict[i].__contains__("MOVELINK"):
    #                 move_links_dict[i] = {role: set() for role in self.link2role_dict["MOVELINK"]}
    #                 move_link_triples.add(("MoveLink", None, i))
    #         if tag in ["B-SPATIAL_SIGNAL_Q", "B-SPATIAL_SIGNAL_Q_O"]:
    #             if not trigger_dict[i].__contains__("QSLINK"):
    #                 t_links_dict[i] = {role: set() for role in self.link2role_dict["QSLINK"]}
    #                 t_link_triples.add(("QSLINK", None, i, None))
    #         if tag in ["B-SPATIAL_SIGNAL_O", "B-SPATIAL_SIGNAL_Q_O"]:
    #             if not trigger_dict[i].__contains__("OLINK"):
    #                 o_links_dict[i] = {role: set() for role in self.link2role_dict["OLINK"]}
    #                 o_link_triples.add(("OLINK", None, i, None))
    #
    #     if decompose:
    #         return move_link_triples, t_link_triples, o_link_triples, no_trigger_triples
    #     else:
    #         return move_links_dict, t_links_dict, o_links_dict, no_trigger_triples

    # def evaluate_exact(self, pred_triples_list):
    #     def calculate_prf(gold, pred, correct):
    #         precision = correct / pred if pred != 0 else 0.0
    #         recall = correct / gold if gold != 0 else 0.0
    #         f_1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0.0
    #         return precision, recall, f_1
    #
    #     link_types = ["MoveLink", "TLINK", "OLINK", "NoTriggerLink", "exact_All"]
    #     eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "exact_p": 0, "exact_r": 0, "exact_f1": 0} for k in
    #                  link_types}
    #
    #     for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
    #         gold_triples = example.triples
    #         gold_links_tuples = self.get_links(gold_triples, decompose=False)
    #         other_gold_links_tuples = self.post_process(gold_triples, example.features["tags"], decompose=False)
    #
    #         pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
    #         pred_links_tuples = self.get_links(pred_triples, decompose=False)
    #         other_pred_links_tuples = self.post_process(pred_triples, example.features["tags"], decompose=False)
    #
    #         for (link_type, gold_links, other_gold_links, pred_links, other_pred_links) in \
    #                 zip(link_types[:-1], gold_links_tuples, other_gold_links_tuples, pred_links_tuples,
    #                     other_pred_links_tuples):
    #             gold_links.update(other_gold_links)
    #             pred_links.update(other_pred_links)
    #
    #             correct_num = 0
    #             if type(gold_links) is dict:
    #                 for trigger, gold_roles in gold_links.items():
    #                     pred_roles = pred_links.get(trigger, {})
    #                     if pred_roles == gold_roles:
    #                         correct_num += 1
    #             else:
    #                 correct_num = len(gold_links & pred_links)
    #
    #             eval_dict[link_type]["predict"] += len(pred_links)
    #             eval_dict[link_type]["gold"] += len(gold_links)
    #             eval_dict[link_type]["correct"] += correct_num
    #
    #             eval_dict["exact_All"]["predict"] += len(pred_links)
    #             eval_dict["exact_All"]["gold"] += len(gold_links)
    #             eval_dict["exact_All"]["correct"] += correct_num
    #
    #     for k, v in eval_dict.items():
    #         p, r, f = calculate_prf(v["gold"], v["predict"], v["correct"])
    #         eval_dict[k]["exact_p"] = p
    #         eval_dict[k]["exact_r"] = r
    #         eval_dict[k]["exact_f1"] = f
    #     return eval_dict

# def convert_examples_to_features(examples, label_list, relation_list, max_seq_length, tokenizer,
#                                  cls_token_at_end=False,
#                                  cls_token='[CLS]',
#                                  cls_token_segment_id=1,
#                                  sep_token='[SEP]',
#                                  sep_token_extra=False,
#                                  pad_on_left=False,
#                                  pad_token=0,
#                                  pad_token_segment_id=0,
#                                  pad_token_label_id=-1,
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
#     relation_map = {tag: i for i, tag in enumerate(relation_list)}
#
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 1000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         tokens, labels = example.tokens, example.labels[:max_seq_length]
#
#         # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
#         special_tokens_count = 3 if sep_token_extra else 2
#         if len(tokens) > max_seq_length - special_tokens_count:
#             tokens = tokens[:(max_seq_length - special_tokens_count)]
#
#         tokens = tokens + [sep_token]
#
#         if sep_token_extra:
#             # roberta uses an extra separator b/w pairs of sentences
#             tokens += [sep_token]
#
#         segment_ids = [sequence_a_segment_id] * len(tokens)
#
#         if cls_token_at_end:
#             tokens = tokens + [cls_token]
#             segment_ids = segment_ids + [cls_token_segment_id]
#         else:
#             tokens = [cls_token] + tokens
#             segment_ids = [cls_token_segment_id] + segment_ids
#
#         label_ids = [label_map[label] for label in labels]
#         label_ids = label_ids + [label_map["O"]] * (max_seq_length - len(label_ids))
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#         padding_length = max_seq_length - len(input_ids)
#
#         if pad_on_left:
#             input_ids = ([pad_token] * padding_length) + input_ids
#             input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
#             segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
#         else:
#             input_ids = input_ids + ([pad_token] * padding_length)
#             input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
#             segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
#
#         selection_matrix = np.zeros((max_seq_length, len(relation_list), max_seq_length), dtype=np.int)
#         none = relation_map["NA"]
#         selection_matrix[:, none, :] = 1
#         for (s, p, o) in example.triples:
#
#             if s > max_seq_length - 1 or o > max_seq_length - 1:
#                 continue
#
#             selection_matrix[s, p, o] = 1
#             selection_matrix[s, none, o] = 0
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#         assert len(label_ids) == max_seq_length
#
#         if ex_index < 5:
#             logger.info("*** Example ***")
#             logger.info("guid: %s" % example.guid)
#             logger.info("tokens: %s" % " ".join(
#                 [str(x) for x in tokens]))
#             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#             logger.info("labels: %s" % " ".join([str(x) for x in labels]))
#             logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
#             logger.info("triples: %s" % " ".join([str(x) for x in example.triples]))
#
#         features.append(
#             InputFeatures(input_ids=input_ids,
#                           input_mask=input_mask,
#                           segment_ids=segment_ids,
#                           label_ids=label_ids,
#                           selection_matrix=selection_matrix))
#     return features


# processors = {
#     "SpaceEval2015": EntityAwareMHSDataProcessor
# }
