# -*- coding: utf-8 -*-
"""
@Date ： 2021/3/28 21:36
@Author ： xyu
"""
import itertools
import logging
import os
from collections import defaultdict, OrderedDict
from enum import Enum
from typing import Tuple, Dict, List
from xml.etree import ElementTree

import numpy as np

from data_utils.entity_aware_mhs_utils import EntityAwareMHSDataProcessor, InputExample, InputFeatures
from utils.bert_utils import align_labels
from utils.seq_labeling_eval import classification_report_dict

logger = logging.getLogger(__name__)


class SpRLTask(Enum):
    TASK_1 = 1
    TASK_2 = 2
    TASK_3 = 3


class SpRLBaseMHSDataProcessor(EntityAwareMHSDataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gold_results = {}
        self.link_types = {}
        # self.role2link_dict = {}
        # self.link2role_dict = {}
        # self.link2rel_type = {}
        self.rel_type2link = {}
        self.task2gold = {}
        self.task2link_types = {
            SpRLTask.TASK_1: ["RELATION"],
            SpRLTask.TASK_2: ["REGION", "DIRECTION", "DISTANCE"],
            SpRLTask.TASK_3: ["REGION", "DIRECTION", "DISTANCE"],
        }
        self.pred_elements = None

    def load_gold_results(self, task):
        if task in self.task2gold:
            return self.task2gold[task]
        role_types = {"SPATIALINDICATOR", "LANDMARK", "TRAJECTOR"}
        path = os.path.join(self.data_dir, "gold.xml")
        tree = ElementTree.parse(path).getroot()
        gold_results = []
        for scene in tree.findall("SCENE"):
            for sentence in scene.findall("SENTENCE"):
                link2set = defaultdict(set)
                element_set = set()
                for element in sentence:
                    if element.tag in role_types and int(element.attrib['start']) != -1:
                        element_set.add(element.attrib['id'])

                for relation in sentence.findall("RELATION"):
                    general_types = relation.attrib["general_type"].split("/")
                    specific_types = relation.attrib["RCC8_value"].split("/")
                    # specific_types = relation.attrib["specific_type"].split("/")

                    trigger = relation.attrib["spatial_indicator_id"]
                    trajector = relation.attrib["trajector_id"]
                    landmark = relation.attrib["landmark_id"]

                    role_ids = tuple(x if x in element_set else "" for x in [trigger, trajector, landmark])

                    for general_type, specific_type in zip(general_types, specific_types):
                        link_type = "RELATION"
                        link_tuple = tuple()
                        link_tuple += role_ids
                        if task != SpRLTask.TASK_1:
                            link_type = general_type.strip()
                            link_tuple += (general_type,)
                        if task == SpRLTask.TASK_3:
                            link_tuple += (specific_type.strip(),)
                        link2set[link_type].add(link_tuple)
                gold_results.append(link2set)
        self.task2gold[task] = gold_results
        return gold_results

    def get_links(self, spo_triples, task: SpRLTask):
        link2tuples = defaultdict(set)
        link_type2rel_types = defaultdict(lambda: defaultdict(set))
        trigger2link_types = defaultdict(set)
        trigger2role_dict = defaultdict(lambda: OrderedDict([("trajector", set()), ("landmark", set())]))

        for (s, p, o) in spo_triples:
            relation = self.relations[p] if isinstance(p, int) else p
            if relation in self.rel_type2link:
                link_type = self.rel_type2link[relation]
                link_type2rel_types[link_type][s].add(relation)
                trigger2link_types[s].add(link_type)
            # elif relation in self.link_types:
            #     trigger2link_types[s].add(relation)
            else:
                trigger2role_dict[s][relation].add(o)

        for trigger, role_dict in trigger2role_dict.items():
            link_types = {''} if task == SpRLTask.TASK_1 else trigger2link_types.get(trigger, {''})
            # link_types = trigger2link_types.get(trigger, {''})
            for link_type in link_types:
                rel_types = link_type2rel_types[link_type].get(trigger, {''})
                # rel_types = {''} if task != SpRLTask.TASK_3 else link_type2rel_types[link_type][trigger]
                if task != SpRLTask.TASK_1:
                    role_dict["general_type"] = {link_type}
                if task == SpRLTask.TASK_3:
                    role_dict["specific_type"] = rel_types
                for role, set_ in role_dict.items():
                    if not set_:
                        set_.add('')
                if link_type == '':
                    link_type = "RELATION"
                for tuple_ in itertools.product(*role_dict.values()):
                    if tuple_[0] == tuple_[1] or tuple_[0] == trigger or tuple_[1] == trigger:
                        # print("sp和tr以及ld不能相同")
                        continue
                    link2tuples[link_type].add((trigger,) + tuple_)
        return link2tuples

    def get_element_ids(self, example, tags, triples):
        element_set = set()
        for s, p, o in triples:
            element_set.add(s)
            element_set.add(o)
        element_ids = example.element_ids
        new_element_ids = [x for x in element_ids]

        for idx in element_set:
            if element_ids[idx] == "":
                new_element_ids[idx] = f"na-{idx}"
                if tags[idx].startswith("B-"):
                    _, _, tag = tags[idx].partition('-')
                    i = idx + 1
                    while i < len(tags) and tags[i] == f"I-{tag}":
                        if element_ids[i] != "":
                            new_element_ids[idx] = element_ids[i]
                            break
                        i += 1
        return new_element_ids

    # def element_id():
    #     if element_ids[idx] != "":
    #         return element_ids[idx]
    #     elif tags[idx].startswith("B-"):
    #         _, _, tag = tags[idx].partition('-')
    #         i = idx + 1
    #         while i < len(tags) and tags[i] == f"I-{tag}":
    #             if element_ids[i] != "":
    #                 return element_ids[i]
    #             i += 1
    #     return f"na-{idx}"


    def evaluate_link(self, pred_triples_list, task: SpRLTask = SpRLTask.TASK_1, strict=True):
        link_types = self.task2link_types[task] + ["OVERALL"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}

        gold_result = self.load_gold_results(task)
        for i, (example, ori_pred_triples, gold_link_dict) in enumerate(zip(self.dev_examples, pred_triples_list, gold_result)):
            element_ids = example.element_ids
            pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
            # pred_triples = example.triples
            # tags = example.features["tags"]

            # for s, p, o in pred_triples:
            #     if element_ids[s] == "":
            #         element_ids[s] = f"na-{s}"

            tags = example.features["tags"] if self.pred_elements is None else self.pred_elements[i]
            element_ids = self.get_element_ids(example, tags, pred_triples)
            pred_triples = [(element_ids[s], p, element_ids[o]) for s, p, o in pred_triples]
            pred_link_dict = self.get_links(pred_triples, task=task)

            for link_type in link_types:
                pred_links = pred_link_dict[link_type]
                gold_links = gold_link_dict[link_type]

                if strict:
                    pred_links = set(filter(lambda x: x[1] != "" and x[2] != "", pred_links))
                    # gold_links = set(filter(lambda x: x[1] != "" and x[2] != "", gold_links))

                correct_links = gold_links & pred_links
                gold_num, pred_num, correct_num = len(gold_links), len(pred_links), len(correct_links)

                eval_dict[link_type]["predict"] += pred_num
                eval_dict[link_type]["gold"] += gold_num
                eval_dict[link_type]["correct"] += correct_num

                if link_type != "NoTrigger":
                    eval_dict["OVERALL"]["predict"] += pred_num
                    eval_dict["OVERALL"]["gold"] += gold_num
                    eval_dict["OVERALL"]["correct"] += correct_num

        for k, v in eval_dict.items():
            p, r, f = self.calculate_prf(v["gold"], v["predict"], v["correct"])
            eval_dict[k]["p"] = p
            eval_dict[k]["r"] = r
            eval_dict[k]["f1"] = f

        return eval_dict

    # def evaluate_rel_types(self, pred_triples_list):
    #     link_types = list(self.link_type2rel_type.keys()) + ["OVERALL"]
    #     eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}
    #
    #     def get_rel_type_dict(triples):
    #         attr_dict = defaultdict(set)
    #         for head, rel, tail in triples:
    #             relation = self.relations[rel] if isinstance(rel, int) else rel
    #             for l_type, rel_types in self.link2role_dict.items():
    #                 if relation in rel_types:
    #                     attr_dict[l_type].add((head, relation, tail))
    #                     break
    #         return attr_dict
    #
    #     for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
    #         pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
    #         gold_dict = get_rel_type_dict(example.triples)
    #         pred_dict = get_rel_type_dict(pred_triples)
    #
    #         for link_type in link_types:
    #             gold_set = gold_dict[link_type]
    #             pred_set = pred_dict[link_type]
    #             correct_set = gold_set & pred_set
    #             gold_num, pred_num, correct_num = len(gold_set), len(pred_set), len(correct_set)
    #
    #             eval_dict[link_type]["predict"] += pred_num
    #             eval_dict[link_type]["gold"] += gold_num
    #             eval_dict[link_type]["correct"] += correct_num
    #
    #             eval_dict["OVERALL"]["predict"] += pred_num
    #             eval_dict["OVERALL"]["gold"] += gold_num
    #             eval_dict["OVERALL"]["correct"] += correct_num
    #
    #     for k, v in eval_dict.items():
    #         p, r, f = self.calculate_prf(v["gold"], v["predict"], v["correct"])
    #         eval_dict[k]["p"] = p
    #         eval_dict[k]["r"] = r
    #         eval_dict[k]["f1"] = f
    #
    #     return eval_dict

    def evaluate_role(self, pred_triples_list):
        role_types = ["trigger", "trajector", "landmark"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in role_types + ["OVERALL"]}

        def get_role_dict(triples):
            attr_dict = defaultdict(set)
            for head, rel, tail in triples:
                relation = self.relations[rel] if isinstance(rel, int) else rel
                if relation in role_types:
                    attr_dict[relation].add(tail)
                    attr_dict["trigger"].add(head)
            return attr_dict

        for i, (example, ori_pred_triples) in enumerate(zip(self.dev_examples, pred_triples_list)):
            pred_triples = self.decode_triples(example.tokens, ori_pred_triples)

            tags = example.features["tags"] if self.pred_elements is None else self.pred_elements[i]
            pred_element_ids = self.get_element_ids(example, tags, pred_triples)
            pred_triples = [(pred_element_ids[s], p, pred_element_ids[o]) for s, p, o in pred_triples]

            element_ids = example.element_ids
            gold_triples = [(element_ids[s], p, element_ids[o]) for s, p, o in example.triples]

            gold_dict = get_role_dict(gold_triples)
            pred_dict = get_role_dict(pred_triples)

            for role_type in role_types:
                gold_set = gold_dict[role_type]
                pred_set = pred_dict[role_type]
                correct_set = gold_set & pred_set
                gold_num, pred_num, correct_num = len(gold_set), len(pred_set), len(correct_set)

                eval_dict[role_type]["predict"] += pred_num
                eval_dict[role_type]["gold"] += gold_num
                eval_dict[role_type]["correct"] += correct_num

                eval_dict["OVERALL"]["predict"] += pred_num
                eval_dict["OVERALL"]["gold"] += gold_num
                eval_dict["OVERALL"]["correct"] += correct_num

        for k, v in eval_dict.items():
            p, r, f = self.calculate_prf(v["gold"], v["predict"], v["correct"])
            eval_dict[k]["p"] = p
            eval_dict[k]["r"] = r
            eval_dict[k]["f1"] = f

        return eval_dict

    def decode_elements(self, tokens, labels):
        align_token_pos = []
        for idx, token in enumerate(tokens):
            sub_tokens = self.tokenizer.tokenize(token)
            align_token_pos.extend([idx] * len(sub_tokens))

        num_tokens = len(tokens)
        new_labels = [''] * num_tokens

        for idx, label in zip(align_token_pos, labels):
            if new_labels[idx] == '':
                new_labels[idx] = label

        return new_labels

    def evaluate_element(self, pred_list: Tuple[Dict], data_type="dev"):
        examples = self.dev_examples if data_type == "dev" else self.test_examples
        # pred_list = list(map(lambda x: x["elements"], pred_result))
        gold_list = list(map(lambda x: x.features["tags"], examples))
        self.pred_elements = pred_list
        pred_labels = list(itertools.chain(*pred_list))
        gold_labels = list(itertools.chain(*gold_list))
        eval_dict = classification_report_dict(gold_labels, pred_labels)
        return eval_dict

    def post_process(self, pred_result, dataset="dev"):
        modified_pred_dict = defaultdict(list)
        examples = self.dev_examples if dataset == "dev" else self.test_examples
        for pred_item, example in zip(pred_result, examples):
            if isinstance(pred_item, dict):
                modified_pred_dict["links"].append(pred_item["links"])
                elements = self.decode_elements(example.tokens, pred_item["elements"])
                modified_pred_dict["elements"].append(elements)
            else:
                modified_pred_dict["links"].append(pred_item)
        return modified_pred_dict

    def get_element_labels(self):
        return self.feature_dict["tags"]

    def save_predict_result(self, output_dir, pred_result, **kwargs):
        super().save_predict_result(output_dir, pred_result["links"])


class SpRLMHSDataProcessTask1(SpRLBaseMHSDataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.link_types = ["RELATION"]
        # self.role2link_dict = {"trajector": "RELATION", "landmark": "RELATION"}
        # self.link2role_dict = {"RELATION": ["trajector", "landmark"]}

    def evaluate(self, pred_result):
        eva_dict = OrderedDict()
        if "elements" in pred_result:
            eva_dict["elements"] = self.evaluate_element(pred_result["elements"])
        links = pred_result["links"]
        eva_dict["role"] = self.evaluate_role(links)
        eva_dict["task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=False)
        eva_dict["strict task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=True)
        return eva_dict


class SpRLMHSDataProcessTask2(SpRLBaseMHSDataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init()
        self.rel_type2link = dict()
        for link, labels in self.link2rel_types.items():
            for label in labels:
                self.rel_type2link[label] = link
        all_rel_types = set(itertools.chain(*self.link2rel_types.values()))
        self.relations = [r for r in self.relations if r not in all_rel_types]
        self.rel2IdMap = {rel: idx for idx, rel in enumerate(self.relations)}
        self.pre_process(self.train_examples)
        self.pre_process(self.dev_examples)
        # self.rel_type_bio_labels = OrderedDict()
        # for link_type, rel_types in self.link2role_dict:
        #     bio_labels = ["O"] + list(sorted(map(lambda x: f"B-{x}", rel_types))) + \
        #                  list(sorted(map(lambda x: f"I-{x}", rel_types)))
        #     self.rel_type_bio_labels[link_type] = bio_labels

        # self.role2link_dict = {"trajector": "RELATION", "landmark": "RELATION"}
        # self.link2role_dict = {
        #     "RELATION": ["trajector", "landmark", "general_type"],
        # }
        # self.attr2label_dict = {
        #     "general_type": ["REGION", "DIRECTION", "DISTANCE"],
        # }

        # self.role2link_dict = {
        #     "trajector_Q": "REGION", "landmark_Q": "REGION",
        #     "trajector_O": "DIRECTION", "landmark_O": "DIRECTION",
        #     "trajector_M": "DISTANCE", "landmark_M": "DISTANCE"
        # }
        # self.link2role_dict = {
        #     "REGION": ["trigger", "trajector_Q", "landmark_Q"],
        #     "DIRECTION": ["trigger", "trajector_O", "landmark_O"],
        #     "DISTANCE": ["trigger", "trajector_M", "landmark_M"]
        # }

    def init(self):
        self.link_types = ["REGION", "DIRECTION", "DISTANCE"]
        self.link2rel_types = {
            "REGION": ["O", "REGION"],
            "DIRECTION": ["O", "DIRECTION"],
            "DISTANCE": ["O", "DISTANCE"]
        }

    def evaluate(self, pred_result):
        eval_dict = OrderedDict()
        if "elements" in pred_result:
            eval_dict["elements"] = self.evaluate_element(pred_result["elements"])
        links = pred_result["links"]
        eval_dict["role"] = self.evaluate_role(links)
        eval_dict["task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=False)
        eval_dict["strict task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=True)
        eval_dict["task2"] = self.evaluate_link(links, task=SpRLTask.TASK_2, strict=False)
        eval_dict["strict task2"] = self.evaluate_link(links, task=SpRLTask.TASK_2, strict=True)
        return eval_dict

    def pre_process(self, examples: List[InputExample]):
        for example in examples:
            # tags = example.features["tags"]
            t_len = len(example.tokens)
            example.rel_types = {k: ["O"] * t_len for k in self.link2rel_types}
            for head, rel, tail in example.triples:
                for link_type, rel_types in self.link2rel_types.items():
                    if rel in rel_types:
                        example.rel_types[link_type][head] = rel
                        # i = head + 1
                        # while i < t_len and tags[i].startswith("I-SPATIAL_SIGNAL"):
                        #     example.rel_types[link_type][i] = "I-" + rel
                        # break

    @staticmethod
    def _post_process_rel_type(pred_tuples):
        pred_triples = []
        for pred_tuple in pred_tuples:
            if len(pred_tuple) == 2:
                idx, label = pred_tuple
                pred_triples.append((idx, label, idx))
                # TODO：是否评测 I-XXX
                # if label.startswith("B-") or label.startswith("I-"):
                #     label = label[2:]
                #     pred_triples.append((idx, label, idx))
            else:
                pred_triples.append(pred_tuple)
        return pred_triples

    def post_process(self, pred_result, **kwargs):
        modified_pred_dict = super().post_process(pred_result, **kwargs)
        links = modified_pred_dict["links"]
        modified_pred_dict["links"] = list(map(self._post_process_rel_type, links))
        return modified_pred_dict

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
                                     max_rel_distance=2,
                                     use_token_pre_idx=True
                                     ):
        sentence_len_dict = defaultdict(int)
        max_length = 0

        rel2id_dict = {label: i for i, label in enumerate(self.relations)}
        features_map = {}
        feature_pad_label = {}
        for k, v in self.feature_dict.items():
            features_map[k] = {item: idx for idx, item in enumerate(v)}
            if features_map[k].__contains__("O"):
                feature_pad_label[k] = features_map[k]["O"]
            else:
                feature_pad_label[k] = features_map[k][0]

        # ******     rel type     ****** #
        rel_type_map = {}
        rel_type_pad_label = {}
        for k, v in self.link2rel_types.items():
            rel_type_map[k] = {item: idx for idx, item in enumerate(v)}
            rel_type_pad_label[k] = rel_type_map[k]["O"]

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 1000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = []

            extra_raw_features = OrderedDict()
            extra_features = OrderedDict()

            raw_rel_types = OrderedDict()
            rel_types = OrderedDict()

            align_token_pos = []
            sub_tokens_list = []

            for i, token in enumerate(example.tokens):
                sub_tokens = tokenizer.tokenize(token)
                tokens.extend(sub_tokens)
                sub_tokens_list.append(sub_tokens)
                align_token_pos.extend([i] * len(sub_tokens))

            for f_name, f_list in example.features.items():
                extra_raw_features[f_name] = align_labels(align_token_pos, f_list)

            for r_name, r_list in example.rel_types.items():
                raw_rel_types[r_name] = align_labels(align_token_pos, r_list)

            if len(tokens) > max_length:
                max_length = len(tokens)

            sentence_len_dict[len(tokens)] += 1

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]

            tokens = [cls_token] + tokens + [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # label_ids = [label_map[label] for label in labels]
            # label_ids = label_ids + [label_map["O"]] * (max_seq_length - len(label_ids))

            # ************ #
            for f_name, f_list in extra_raw_features.items():
                f_list = [features_map[f_name][item] for item in f_list]
                f_pad_label = feature_pad_label[f_name]
                if ignore_cls_sep:
                    f_list = f_list[:max_seq_length]
                else:
                    f_list = [f_pad_label] + f_list[:max_seq_length - 2] + [f_pad_label]
                f_list = f_list + [f_pad_label] * (max_seq_length - len(f_list))
                extra_features[f_name] = f_list
            # ************ #
            for link_type, r_list in raw_rel_types.items():
                r_list = [rel_type_map[link_type][item] for item in r_list]
                r_pad_label = rel_type_pad_label[link_type]
                if ignore_cls_sep:
                    r_list = r_list[:max_seq_length]
                else:
                    r_list = [r_pad_label] + r_list[:max_seq_length - 2] + [r_pad_label]
                r_list = r_list + [r_pad_label] * (max_seq_length - len(r_list))
                rel_types[link_type] = r_list
            # ************ #

            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)

            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            selection_matrix = np.zeros((max_seq_length, len(self.relations), max_seq_length), dtype=np.int)
            none = rel2id_dict["NA"]
            selection_matrix[:, none, :] = 1
            selection_triples = self.align_selection_triple(align_token_pos, example.triples)
            for (s, p, o) in selection_triples:
                if s >= max_seq_length or o >= max_seq_length:
                    continue
                if p not in self.rel2IdMap:
                    continue
                p = self.rel2IdMap[p]
                selection_matrix[s, p, o] = 1
                selection_matrix[s, none, o] = 0
            relative_positions, entity_mask = None, None

            token_to_ori_idx = None
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
                if use_token_pre_idx:
                    length = max_seq_length - len(sub_token_to_ori_idx)
                    token_to_ori_idx = sub_token_to_ori_idx[:max_seq_length] + [max_seq_length] * length

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
                              rel_types=rel_types,
                              token_to_pre_idx=token_to_ori_idx))

        print("max_length", max_length)
        print(list(sorted(sentence_len_dict.items(), key=lambda x: x[0])))
        return features


class SpRLMHSDataProcessTask3_1(SpRLBaseMHSDataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.link_types = ["REGION", "DIRECTION", "DISTANCE"]
        self.link2rel_types = {
            "REGION": ["O", "DC", "EC", "EQ", "NTPP", "NTPPI", "NTTP", "PO", "TPP", "TPPI"],
            "DIRECTION": ["O", "ABOVE", "BEHIND", "BELOW", "FRONT", "LEFT", "RIGHT"],
            "DISTANCE": ["O", "10METERS", "CLOSE", "FAR", "LAST", "MIDDLE", "NEAR", "SECOND"]
        }
        self.rel_type2link = dict()
        for link, labels in self.link2rel_types.items():
            for label in labels:
                self.rel_type2link[label] = link

        # self.role2link_dict = {"trajector": "RELATION", "landmark": "RELATION"}
        # self.link2role_dict = {
        #     "RELATION": ["trajector", "landmark", "general_type", "specific_type"],
        # }
        #
        # self.rel_type_dict = {
        #     "REGION": {"DC", "EC", "EQ", "NTPP", "NTPPI", "NTTP", "PO", "TPP", "TPPI"},
        #     "DIRECTION": {"ABOVE", "BEHIND", "BELOW", "FRONT", "LEFT", "RIGHT"},
        #     "DISTANCE": {"10METERS", "CLOSE", "FAR", "LAST", "MIDDLE", "NEAR", "SECOND"}
        # }
        #
        # self.attr2label_dict = {
        #     "general_type": {"REGION", "DIRECTION", "DISTANCE"},
        #     "specific_type": set(itertools.chain(*self.rel_type_dict.values()))
        # }

    def evaluate(self, pred_result):
        eval_dict = OrderedDict()
        if "elements" in pred_result:
            eval_dict["elements"] = self.evaluate_element(pred_result["elements"])
        links = pred_result["links"]
        eval_dict["role"] = self.evaluate_role(links)
        eval_dict["task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=False)
        eval_dict["strict task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=True)
        eval_dict["task2"] = self.evaluate_link(links, task=SpRLTask.TASK_2, strict=False)
        eval_dict["strict task2"] = self.evaluate_link(links, task=SpRLTask.TASK_2, strict=True)
        eval_dict["task3"] = self.evaluate_link(links, task=SpRLTask.TASK_3, strict=False)
        eval_dict["strict task3"] = self.evaluate_link(links, task=SpRLTask.TASK_3, strict=True)
        return eval_dict


class SpRLMHSDataProcessTask3_2(SpRLMHSDataProcessTask2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        self.link_types = ["REGION", "DIRECTION", "DISTANCE"]
        self.link2rel_types = {
            "REGION": ["O", "DC", "EC", "EQ", "NTPP", "NTPPI", "NTTP", "PO", "TPP", "TPPI"],
            "DIRECTION": ["O", "ABOVE", "BEHIND", "BELOW", "FRONT", "LEFT", "RIGHT"],
            "DISTANCE": ["O", "10  METERS", "SECOND  ROW", "CLOSE", "FAR", "LAST", "MIDDLE", "NEAR", "SECOND"]
        }

    def evaluate(self, pred_result):
        eval_dict = OrderedDict()
        if "elements" in pred_result:
            eval_dict["elements"] = self.evaluate_element(pred_result["elements"])
        links = pred_result["links"]
        eval_dict["role"] = self.evaluate_role(links)
        eval_dict["task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=False)
        eval_dict["strict task1"] = self.evaluate_link(links, task=SpRLTask.TASK_1, strict=True)
        eval_dict["task2"] = self.evaluate_link(links, task=SpRLTask.TASK_2, strict=False)
        eval_dict["strict task2"] = self.evaluate_link(links, task=SpRLTask.TASK_2, strict=True)
        eval_dict["task3"] = self.evaluate_link(links, task=SpRLTask.TASK_3, strict=False)
        eval_dict["strict task3"] = self.evaluate_link(links, task=SpRLTask.TASK_3, strict=True)
        return eval_dict


