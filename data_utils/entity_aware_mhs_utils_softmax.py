import itertools
import logging
from collections import OrderedDict, defaultdict

import numpy as np
import os

from data_utils.entity_aware_dep_mhs_utils import InputFeatures
from data_utils.entity_aware_mhs_utils import EntityAwareMHSDataProcessor_2, InputExample
from utils.bert_utils import align_labels
from utils.spaceeval_utils_new import load_links_from_file
from utils.spaceeval_utils import Metrics

logger = logging.getLogger(__name__)


class SoftmaxEntityAwareMHSDataProcessor(EntityAwareMHSDataProcessor_2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_signals = ["B-SPATIAL_SIGNAL_Q", "I-SPATIAL_SIGNAL_Q", "B-SPATIAL_SIGNAL_O", "I-SPATIAL_SIGNAL_O"]
        self.ss2linkType = {
            "SPATIAL_SIGNAL": ["QSLINK", "OLINK"],
            "SPATIAL_SIGNAL_Q": ["QSLINK"],
            "SPATIAL_SIGNAL_O": ["OLINK"],
        }

        self.role2link_dict = {"trajector": "LINK", "landmark": "LINK", "locatedIn": "NoTrigger",
                               "mover": "MOVELINK", "source": "MOVELINK", "midPoint": "MOVELINK", "goal": "MOVELINK",
                               "motion_signalID": "MOVELINK", "pathID": "MOVELINK", "ground": "MOVELINK"}
        self.link2role_dict = {
            "QSLINK": ["trajector", "landmark"],
            "OLINK": ["trajector", "landmark"],
            "MOVELINK": ["mover", "source", "goal", "ground", "midPoint", "pathID", "motion_signalID"],
            "NoTrigger": ["locatedIn"],
        }

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
        features_map = {}
        feature_pad_label = {}
        for k, v in self.feature_dict.items():
            features_map[k] = {item: idx for idx, item in enumerate(v)}
            for label in self.spatial_signals:
                if label in features_map[k]:
                    features_map[k][label] = features_map[k][label[:-2]]
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

            selection_matrix = np.zeros((max_seq_length, max_seq_length), dtype=np.int)
            none_label_id = rel2idMap["NA"]
            selection_matrix[:] = none_label_id
            selection_triples = self.align_selection_triple(align_token_pos, example.triples)
            for (s, p, o) in selection_triples:
                if s > max_seq_length - 1 or o > max_seq_length - 1:
                    continue
                selection_matrix[s, o] = p

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
                              selection_matrix=selection_matrix))
        print("max_length", max_length)
        return features

    def get_links(self, spo_triples, example: InputExample):
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
                continue

            if link_type == "LINK":
                tag = example.features["tags"][s][2:]
                link_types = self.ss2linkType.get(tag, [])
            else:
                link_types = [link_type]
            if example.element_ids:
                s, o = example.element_ids[s], example.element_ids[o]
            for link_type in link_types:
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


    def evaluate(self, pred_triples_list, allow_null_mover=True):
        link_types = ["MOVELINK", "QSLINK", "OLINK", "NoTrigger", "OVERALL"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}
        # TODO:
        # eval_optional_roles = "source" in self.relations
        for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
            gold_triples = example.triples
            gold_link_dict = self.get_links(gold_triples, example)
            other_gold_link_dict = self.post_process(gold_triples, example.features["tags"])

            pred_triples = self.decode_triples(example.tokens, ori_pred_triples)
            pred_link_dict = self.get_links(pred_triples, example)
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

    def evaluate_1(self, pred_triples_list, metric: Metrics, eval_optional_roles=True):
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
            # pred_triples = [(element_ids[s], p, element_ids[o]) for s, p, o in pred_triples]
            pred_link_dict = self.get_links(pred_triples, example)
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
            gold_link_dict = load_links_from_file(path, metric, eval_optional_roles, eval_null_mover)

            # pred_link_dict["QSLINK"].extend(pred_link_dict["NoTrigger"])

            for link_type in link_types:
                gold_links = gold_link_dict[link_type]
                pred_links = pred_link_dict[link_type]

                correct_links = gold_links & pred_links
                gold_num, pred_num, correct_num = len(gold_links), len(pred_links), len(correct_links)

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
