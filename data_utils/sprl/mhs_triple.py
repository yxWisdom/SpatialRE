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
from xml.dom import minidom
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
        self.rel_type2link = {}
        self.task2gold = {}
        self.task2link_types = {
            SpRLTask.TASK_1: ["RELATION"],
            SpRLTask.TASK_2: ["REGION", "DIRECTION", "DISTANCE"],
            SpRLTask.TASK_3: ["REGION", "DIRECTION", "DISTANCE"],
        }
        self.pred_elements = None

    def load_dataset(self, path, task):
        key = path + " " + str(task)
        if key in self.task2gold:
            return self.task2gold[key]
        role_types = {"SPATIAL_INDICATOR", "TRAJECTOR", "LANDMARK"}
        tree = ElementTree.parse(path).getroot()
        gold_results = []
        for sentence in tree.findall("SENTENCE"):
            link2set = defaultdict(set)
            element_set = set()
            for element in sentence:
                if element.tag in role_types and element.text.strip() != "undefined":
                    element_set.add(element.attrib['id'][1:])

            for relation in sentence.findall("RELATION"):
                general_type = relation.attrib["general_type"].upper().strip()
                trigger = relation.attrib["sp"][1:]
                trajector = relation.attrib["tr"][1:]
                landmark = relation.attrib["lm"][1:]

                role_ids = tuple(x if x in element_set else "" for x in [trigger, trajector, landmark])

                link_type = "RELATION"
                link_tuple = role_ids

                if task != SpRLTask.TASK_1:
                    link_type = general_type
                    link_tuple += (general_type,)

                link2set[link_type].add(link_tuple)

            gold_results.append(link2set)
        self.task2gold[key] = gold_results
        return gold_results

    def load_gold_dataset(self, task):
        path = os.path.join(self.data_dir, "gold.xml")
        return self.load_dataset(path, task)

    def load_train_dataset(self, task):
        path = os.path.join(self.data_dir, "train.xml")
        return self.load_dataset(path, task)

    def get_links(self, relations, task: SpRLTask):
        link2tuples = defaultdict(set)
        for relation in relations:
            if task == SpRLTask.TASK_1:
                link2tuples["RELATION"].add(relation[:3])
            elif task == SpRLTask.TASK_2:
                link2tuples[relation[-1]].add(relation)
            else:
                rel_type = relation[-1]
                link_type = self.rel_type2link[rel_type]
                link2tuples[link_type].add(relation[:3] + (link_type, rel_type))
        return link2tuples

    def evaluate_link(self, pred_relations_list, task: SpRLTask = SpRLTask.TASK_1, strict=True):
        link_types = self.task2link_types[task] + ["OVERALL"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}
        gold_data = self.load_gold_dataset(task)
        for pred_relations, gold_link_dict in zip(pred_relations_list, gold_data):
            pred_link_dict = self.get_links(pred_relations, task=task)
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

    def evaluate_role(self, pred_relations_list):
        def get_role_dict(relations):
            attr_dict = defaultdict(set)
            for relation in relations:
                for role_type, role_id in zip(role_types, relation):
                    attr_dict[role_type].add(role_id)
            return attr_dict

        role_types = ["trigger", "trajector", "landmark"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in
                     role_types + ["OVERALL"]}

        gold_data = self.load_gold_dataset(task=SpRLTask.TASK_1)
        for pred_relations, gold_link_dict in zip(pred_relations_list, gold_data):

            gold_dict = get_role_dict(pred_relations)
            pred_dict = get_role_dict(gold_link_dict["RELATION"])

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

    def get_train_data(self, task):
        train_data = []
        train_relations = self.load_train_dataset(task)
        max_len = 0
        for train_item in train_relations:
            dict_ = defaultdict(dict)
            for link_type, tuples in train_item.items():
                for tuple_ in tuples:
                    triple = tuple_[:3]
                    if task != SpRLTask.TASK_1:
                        dict_[link_type][triple] = link_type
                    else:
                        dict_[link_type][triple] = tuple_[-1]
                    max_len = max(max_len, len(tuples))
            train_data.append(dict_)

        return train_data

    def get_element_ids(self, dataset="train"):
        if dataset == "train":
            examples = self.train_examples
        elif dataset == "dev":
            examples = self.dev_examples
        else:
            examples = self.test_examples

        return list(map(lambda x: x.element_ids, examples))

    def save_predict_result(self, output_dir, pred_result, **kwargs):
        self.save_xml(output_dir, pred_result)
        # triples_list = []
        # for links in pred_result["links"]:
        #     triples = []
        #     for link in links:
        #         new_link = list(map(lambda x: ''.join(filter(str.isdigit, x)), link[:3]))
        #         if link[1] != "":
        #             triples.append((int(new_link[0]), "trajector", int(new_link[1])))
        #         if link[2] != "":
        #             triples.append((int(new_link[0]), "landmark", int(new_link[2])))
        #     triples_list.append(triples)
        # pred_result["links"] = triples_list
        # if "elements" not in pred_result:
        #     super().save_predict_result(output_dir, pred_result["links"])
        # else:
        #     self.save_joint_predict_result(output_dir, pred_result, **kwargs)


    def save_xml(self, output_dir, pred_result):
        ele_types = ["SPATIAL_INDICATOR", "TRAJECTOR", "LANDMARK"]
        role_types = ["sp", "tr", "lm"]
        path = os.path.join(self.data_dir, "gold.xml")
        tree = ElementTree.parse(path)
        root = tree.getroot()
        for sentence, pred_triples in zip(root.findall("SENTENCE"), pred_result["links"]):
            sentence_text = ""
            all_elements = [element for element in sentence]
            for element in all_elements:
                if element.tag != "CONTENT":
                    sentence.remove(element)
                else:
                    sentence_text = element.text
            tokens = sentence_text.split(" ")
            sen_len = len(tokens)
            pred_link_dict = self.get_links(pred_triples, task=SpRLTask.TASK_2)
            element_dict = defaultdict(set)
            relations = []

            for rel_type, links in pred_link_dict.items():
                for link in links:
                    relation = ElementTree.Element('RELATION')
                    relation.set("id", f"r{len(relations)}")
                    for ele_type, role_type, role_idx in zip(ele_types, role_types, link[:3]):
                        if role_idx == "":
                            role_idx = sen_len
                            text = "undefined"
                        else:
                            role_idx = int(''.join(filter(str.isdigit, role_idx)))
                            if role_idx >= len(tokens):
                                print(1)
                            text = tokens[role_idx]
                        role_id = f"{role_type[0]}w{role_idx}"
                        relation.set(role_type, role_id)
                        element_dict[ele_type].add((role_id, text))
                    relation.set("general_type", rel_type.lower())
                    relations.append(relation)

            for ele_type, elements in element_dict.items():
                for (role_id, text) in elements:
                    element = ElementTree.Element(ele_type)
                    element.set('id', role_id)
                    element.text = text
                    sentence.append(element)
            for relation in relations:
                sentence.append(relation)

        xml_str = minidom.parseString(ElementTree.tostring(root)).toprettyxml(indent="   ")
        output_path = os.path.join(output_dir, "predict.xml")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

    def save_joint_predict_result(self, output_dir, pred_result, suffix="", include_gold=True, mark_error=True):
        lines = []
        for example, pred_elements, ori_pred_triples in zip(self.dev_examples, pred_result["elements"], pred_result["links"]):
            tokens, features = example.tokens, example.features

            gold_elements = features["tags"]

            pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
            gold_triples = example.triples

            gold_relations, gold_heads = [[] for _ in tokens], [[] for _ in tokens]
            pred_relations, pred_heads = [[] for _ in tokens], [[] for _ in tokens]

            for (s, p, o) in gold_triples:
                gold_relations[s].append(self.relations[p] if isinstance(p, int) else p)
                gold_heads[s].append(o)
            for (s, p, o) in pred_triples:
                pred_relations[s].append(self.relations[p] if isinstance(p, int) else p)
                pred_heads[s].append(o)

            for i, (token, gold_element, gold_relation, gold_head, pred_element, pred_relation, pred_head) in \
                    enumerate(zip(
                        tokens, gold_elements, gold_relations, gold_heads, pred_elements, pred_relations, pred_heads)):

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

                line = f'{i}\t{token.ljust(15)}\t{gold_element.ljust(16)}\t'
                # line = "{i}\t{token}\t{label}\t".format(token=token.ljust(15), label=label.ljust(16))
                if include_gold:
                    gold_relation_str = "[{}]".format(",".join(gold_relation))
                    line += f"{gold_relation_str.ljust(20)}\t{str(list(gold_head)).ljust(10)}\t"

                pred_relation_str = "[{}]".format(",".join(pred_relation))
                line += f"{pred_element.ljust(16)}\t{pred_relation_str.ljust(20)}\t{str(list(pred_head)).ljust(10)}\t"

                if mark_error and gold_element != pred_element:
                    line += "Element Error "

                if mark_error and (pred_head != gold_head or pred_relation != gold_relation):
                    line += "Link Error"

                lines.append(line)

            lines.append("")
        output_path = os.path.join(output_dir, f"predict{suffix}.txt")
        with open(output_path, "w", encoding="utf-8") as file:
            for line in lines:
                file.write(line + "\n")


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


class SpRLMHSDataProcessTask1_softmax(SpRLMHSDataProcessTask1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_examples_to_features(self, *args, **kwargs):
        kwargs['multi_label'] = False
        return super().convert_examples_to_features(*args, **kwargs)


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

    def get_train_data(self, *args, **kwargs):
        return super().get_train_data(SpRLTask.TASK_2)

    def pre_process(self, examples: List[InputExample]):
        for example in examples:
            # tags = example.features["tags"]
            t_len = len(example.tokens)
            example.rel_types = {k: ["O"] * t_len for k in self.link2rel_types}
            for head, rel, tail in example.triples:
                for link_type, rel_types in self.link2rel_types.items():
                    if rel in rel_types:
                        example.rel_types[link_type][head] = rel



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
                                     use_token_pre_idx=True,
                                     multi_label=True
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

            selection_triples = self.align_selection_triple(align_token_pos, example.triples)
            none = rel2id_dict["NA"]
            if multi_label:
                selection_matrix = np.zeros((max_seq_length, len(self.relations), max_seq_length), dtype=np.int)
                selection_matrix[:, none, :] = 1
            else:
                selection_matrix = np.zeros((max_seq_length, max_seq_length), dtype=np.int)
                selection_matrix[:] = none

            for (s, p, o) in selection_triples:
                if s >= max_seq_length or o >= max_seq_length:
                    continue
                if p not in self.rel2IdMap:
                    continue
                p = self.rel2IdMap[p]
                if multi_label:
                    selection_matrix[s, p, o] = 1
                    selection_matrix[s, none, o] = 0
                else:
                    selection_matrix[s, o] = p
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


class SpRLMHSDataProcessTask2_softmax(SpRLMHSDataProcessTask2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_examples_to_features(self, *args, **kwargs):
        kwargs['multi_label'] = False
        return super().convert_examples_to_features(*args, **kwargs)
