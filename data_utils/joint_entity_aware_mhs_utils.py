import itertools
import os
from typing import Tuple, Dict

from data_utils.entity_aware_mhs_utils import EntityAwareMHSDataProcessor_3, EntityAwareMHSDataProcessor_4
from utils.seq_labeling_eval import classification_report_dict


class BaseConfig1DataProcessor:
    def __init__(self, *args, **kwargs):
        pass

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

    def evaluate_element(self, pred_result: Tuple[Dict], data_type="dev", eval_full=False):

        def process_ss(x):
            if x.endswith("_O"):
                x = x[:-2]
            if x.endswith("_Q"):
                x = x[:-2]
            return x

        examples = self.dev_examples if data_type == "dev" else self.test_examples
        pred_list = list(map(lambda x: x["elements"], pred_result))
        gold_list = list(map(lambda x: x.features["tags"], examples))
        pred_labels = list(itertools.chain(*pred_list))
        gold_labels = list(itertools.chain(*gold_list))

        gold_labels = list(map(process_ss, gold_labels))
        pred_labels = list(map(process_ss, pred_labels))

        if not eval_full:
            eval_labels = ["PLACE", "PATH", "NONMOTION_EVENT", "SPATIAL_ENTITY", "MOTION"]
        else:
            eval_labels = None
        eval_dict = classification_report_dict(gold_labels, pred_labels, eval_labels=eval_labels)
        return eval_dict

    def save_predict_result(self, output_dir, pred_result, suffix="", include_gold=True, mark_error=True):
        lines = []
        for example, pred_dict in zip(self.dev_examples, pred_result):
            tokens, features = example.tokens, example.features

            ori_pred_elements = pred_dict['elements']
            pred_elements = self.decode_elements(tokens, ori_pred_elements)
            gold_elements = features["tags"]

            ori_pred_triples = pred_dict['links']
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
                line += "{}\t{}\t".format(pred_relation_str.ljust(20), str(list(pred_head)).ljust(10))

                if mark_error and gold_element != pred_element:
                    line += "Element Error"

                if mark_error and (pred_head != gold_head or pred_relation != gold_relation):
                    line += "Link Error"

                lines.append(line)

            lines.append("")
        output_path = os.path.join(output_dir, f"predict{suffix}.txt")
        with open(output_path, "w", encoding="utf-8") as file:
            for line in lines:
                file.write(line + "\n")


# 用于SpaceEval config1任务，即同时识别元素和关系
class SpaceEvalConfig1DataProcessor_1(EntityAwareMHSDataProcessor_3, BaseConfig1DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_link_attribute(self, pred_result):
        pred_triples_list = list(map(lambda x: x["links"], pred_result))
        return super().evaluate_link_attribute(pred_triples_list)

    def evaluate_exact(self, pred_result, metric, eval_optional_roles=True, eval_link_attr=True):
        pred_triples_list = list(map(lambda x: x["links"], pred_result))
        return super().evaluate_exact(pred_triples_list, metric, eval_optional_roles, eval_link_attr)

    def convert_examples_to_features(self, *args, **kwargs):
        return super().convert_examples_to_features(*args, use_token_pre_idx=True, **kwargs)

    def save_predict_result(self, output_dir, pred_result, **kwargs):
        BaseConfig1DataProcessor.save_predict_result(self, output_dir, pred_result, **kwargs)

    def post_process(self, pred_result):
        for pred_dict, example in zip(pred_result, self.dev_examples):
            pred_dict["elements"] = super().decode_elements(example.tokens, pred_dict["elements"])
        return pred_result


class SpaceEvalConfig1DataProcessor_2(EntityAwareMHSDataProcessor_4, BaseConfig1DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_link_attribute(self, pred_result):
        pred_triples_list = list(map(lambda x: x["links"], pred_result))
        return super().evaluate_link_attribute(pred_triples_list)

    def evaluate_exact(self, pred_result, metric, eval_optional_roles=True, eval_link_attr=True):
        pred_triples_list = list(map(lambda x: x["links"], pred_result))
        return super().evaluate_exact(pred_triples_list, metric, eval_optional_roles, eval_link_attr)

    def convert_examples_to_features(self, *args, **kwargs):
        return super().convert_examples_to_features(*args, use_token_pre_idx=True, **kwargs)

    def save_predict_result(self, output_dir, pred_result, **kwargs):
        BaseConfig1DataProcessor.save_predict_result(self, output_dir, pred_result, **kwargs)

    def post_process(self, pred_result):
        self._add_rel_type_to_triples()
        for pred_dict, example in zip(pred_result, self.dev_examples):
            pred_dict["elements"] = super().decode_elements(example.tokens, pred_dict["elements"])
            pred_dict["links"] = super()._post_process_rel_type(pred_dict["links"])
        return pred_result

