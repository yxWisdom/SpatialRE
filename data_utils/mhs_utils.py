import logging
import operator
import os
import numpy as np
from itertools import groupby

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text, tokens=None, labels=None, triples=None):
        self.guid = guid
        self.text = text
        self.tokens = tokens
        self.labels = labels
        self.triples = triples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, selection_matrix):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.selection_matrix = selection_matrix


class MultiHeadSelectionDataProcessor(object):
    def __init__(self, data_dir=None, tokenizer=None, use_head=True):
        self.labels = None
        self.relations = None
        self.rel2IdMap = None
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        self.use_head = use_head
        # self.pad_label = "PAD"

        self.init()

        self.train_examples = self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")
        self.dev_examples = self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")
        self.test_examples = self._create_example(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_train_examples(self, data_dir):
        return self.train_examples

    def get_dev_examples(self, data_dir):
        return self.dev_examples

    def get_test_examples(self, data_dir):
        return self.test_examples

    def get_labels(self):
        return self.labels

    def get_relations(self):
        return self.relations

    def init(self):
        label_set = set()
        relation_set = set()
        examples = self._read_data(os.path.join(self.data_dir, "train.txt"))
        for example in examples:
            _, _, labels, relations_list, _ = (list(field) for field in zip(*example))
            for relations in relations_list:
                relations = relations[1:-1].replace(" ", "").split(",")
                relation_set.update(relations)
            label_set.update(labels)

        self.labels = sorted(sorted(label_set, key=lambda x: x[:1]), key=lambda x: x[1:])
        self.relations = sorted(relation_set)

        self.rel2IdMap = {relation: idx for idx, relation in enumerate(self.relations)}

    @staticmethod
    def _read_data(input_file, delimiter='\t'):
        raw_examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for is_divider, group_lines in groupby(f, lambda x: x.strip() == ''):
                if not is_divider:
                    raw_examples.append(
                        [map(lambda x: x.strip(), line.strip().split(delimiter)) for line in group_lines])
        return raw_examples

    def trans_to_bert(self, tokens, labels, triples):
        def head2new(old_head):
            if self.use_head:
                return align_token_pos.index(old_head)
            else:
                return len(align_token_pos) - list(reversed(align_token_pos)).index(old_head) - 1

        align_token_pos, bert_tokens, align_labels, new_triples = [], [], [], []
        token_pieces_list = [self.tokenizer.tokenize(token) for token in tokens]
        # align position
        for i, token_pieces in enumerate(token_pieces_list):
            align_token_pos.extend([i] * len(token_pieces))
            bert_tokens.extend(token_pieces)

        for i, pos in enumerate(align_token_pos):
            if i > 0 and pos == align_token_pos[i - 1] and labels[pos].startswith("B-"):
                align_labels.append("I-" + labels[pos][2:])
            else:
                align_labels.append(labels[pos])

        for triple in triples:
            new_triples.append((head2new(triple[0]), triple[1], head2new(triple[2])))

        return bert_tokens, align_labels, new_triples

    @staticmethod
    def trans_from_bert(tokens, labels, triples):
        new_tokens, new_labels, new_triples, align_token_pos = [], [], [], []
        for token, label in zip(tokens, labels):
            if token.startswith("##"):
                new_tokens[-1] += token[2:]
            else:
                new_tokens.append(token)
                new_labels.append(label)
            align_token_pos.append(len(new_tokens) - 1)

        for (s, p, o) in triples:
            new_triples.append((align_token_pos[s], p, align_token_pos[o]))
        return new_tokens, new_labels, new_triples

    def _create_example(self, raw_examples, set_type):
        examples = []
        for i, raw_example in enumerate(raw_examples):
            guid = "%s-%s" % (set_type, i)
            indexes, tokens, labels, relations_list, heads_list = (list(field) for field in zip(*raw_example))
            text = " ".join(tokens)
            triples = []
            for idx, (relations, heads) in enumerate(zip(relations_list, heads_list)):
                relations = relations[1:-1].replace(" ", "").split(",")
                heads = list(map(int, heads[1:-1].split(",")))
                for (relation, head) in zip(relations, heads):
                    if 'NA' not in relations:
                        triples.append((idx, self.relations.index(relation), int(head)))

            tokens, labels, triples = self.trans_to_bert(tokens, labels, triples)
            examples.append(InputExample(guid=guid, text=text, tokens=tokens, labels=labels, triples=triples))
        return examples

    def save_predict_result(self, output_dir, pred_triples_list, include_gold=True, mark_error=True):
        lines = []
        for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
            tokens, labels, gold_triples = self.trans_from_bert(example.tokens, example.labels, example.triples)
            _, _, pred_triples = self.trans_from_bert(example.tokens, example.labels, ori_pred_triples)
            gold_relations, gold_heads = [[] for _ in tokens], [[] for _ in tokens]
            pred_relations, pred_heads = [[] for _ in tokens], [[] for _ in tokens]
            for (s, p, o) in gold_triples:
                gold_relations[s].append(self.relations[p])
                gold_heads[s].append(o)
            for (s, p, o) in pred_triples:
                pred_relations[s].append(self.relations[p])
                pred_heads[s].append(o)
            for i, (token, label, gold_relation, gold_head, pred_relation, pred_head) in \
                    enumerate(zip(tokens, labels, gold_relations, gold_heads, pred_relations, pred_heads)):
                gold_relation = ["NA"] if len(gold_relation) == 0 else gold_relation
                gold_head = [i] if len(gold_head) == 0 else gold_head
                pred_relation = ["NA"] if len(pred_relation) == 0 else pred_relation
                pred_head = [i] if len(pred_head) == 0 else pred_head

                gold_head = sorted(gold_head)
                pred_head = sorted(pred_head)
                gold_relation = sorted(gold_relation)
                pred_relation = sorted(pred_relation)

                line = "{token}\t{label}\t".format(token=token.ljust(15), label=label.ljust(16))
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

    def get_links(self, spo_triples):
        mover_type = "mover"
        trajector_type = "trajector"
        landmark_type = "landmark"
        trigger_type = "trigger"
        located_in_type = "locatedIn"

        move_links = {}
        non_move_links = {}

        move_link_triples = set()
        non_move_link_tuples = set()
        no_trigger_triples = set()

        for (s, p, o) in spo_triples:
            if p in [self.rel2IdMap[trajector_type], self.rel2IdMap[landmark_type]]:
                if not non_move_links.__contains__(o):
                    non_move_links[o] = {trajector_type: [], "trigger": o, landmark_type: []}
                non_move_links[o][self.relations[p]].append(s)
            elif p == self.rel2IdMap[mover_type]:
                if not move_links.__contains__(o):
                    move_links[o] = {mover_type: [], trigger_type: o}
                move_links[o][self.relations[p]].append(s)
            elif p == self.rel2IdMap[located_in_type]:
                no_trigger_triples.add(("NoTriggerLink", s, o))

            # 处理逆关系
            # elif p == self.rel2IdMap[located_in_type]

        for trigger, link in move_links.items():
            movers = link[mover_type] if len(link[mover_type]) else [None]
            for mover in movers:
                move_link_triples.add(("MoveLink", mover, trigger))
        for trigger, link in non_move_links.items():
            trajectors = link[trajector_type] if len(link[trajector_type]) > 0 else [None]
            landmarks = link[landmark_type] if len(link[landmark_type]) > 0 else [None]
            for trajector in trajectors:
                for landmark in landmarks:
                    non_move_link_tuples.add(("NonMoveLink", trajector, trigger, landmark))

        return move_link_triples, non_move_link_tuples, no_trigger_triples

    def evaluation(self, pred_triples_list):
        def calculate_prf(gold, pred, correct):
            precision = correct / pred if pred != 0 else 0.0
            recall = correct / gold if gold != 0 else 0.0
            f_1 = 2 * precision * recall / (precision + recall) if precision !=0 or recall != 0 else 0.0
            return precision, recall, f_1

        link_types = ["MoveLink", "NonMoveLink", "NoTriggerLink", "All"]
        eval_dict = {k: {"predict": 0, "gold": 0, "correct": 0, "p": 0, "r": 0, "f1": 0} for k in link_types}

        for example, ori_pred_triples in zip(self.dev_examples, pred_triples_list):
            _, _, gold_triples = self.trans_from_bert(example.tokens, example.labels, example.triples)
            _, _, pred_triples = self.trans_from_bert(example.tokens, example.labels, ori_pred_triples)
            gold_links_tuples = self.get_links(gold_triples)
            pred_links_tuples = self.get_links(pred_triples)

            for (link_type, gold_links, pred_links) in zip(link_types[:-1], gold_links_tuples, pred_links_tuples):
                eval_dict[link_type]["predict"] += len(pred_links)
                eval_dict[link_type]["gold"] += len(gold_links)
                eval_dict[link_type]["correct"] += len(gold_links & pred_links)

                eval_dict["All"]["predict"] += len(pred_links)
                eval_dict["All"]["gold"] += len(gold_links)
                eval_dict["All"]["correct"] += len(gold_links & pred_links)

        for k, v in eval_dict.items():
            p, r, f = calculate_prf(v["gold"], v["predict"], v["correct"])
            eval_dict[k]["p"] = p
            eval_dict[k]["r"] = r
            eval_dict[k]["f1"] = f

        return eval_dict

        # gold_m_links, gold_qs_links, gold_no_trigger_links = self.get_links(example.triples)
        # pred_m_links, pred_qs_links, pred_no_trigger_links = self.get_links(pred_triples)
        #
        # m_pred, m_gold, m_correct = len(pred_m_links), len(gold_m_links), len(pred_m_links & gold_m_links)
        #
        # qs_pred, qs_gold, move_correct = len(pred_qs_links), len(gold_qs_links), len(pred_qs_links & gold_qs_links)
        #
        # no_prd, no_gold, no_correct = len(pred_no_trigger_links), len(gold_no_trigger_links), \
        #                               len(pred_no_trigger_links & gold_no_trigger_links)


def convert_examples_to_features(examples, label_list, relation_list, max_seq_length, tokenizer,
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
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    relation_map = {tag: i for i, tag in enumerate(relation_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens, labels = example.tokens, example.labels[:max_seq_length]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]

        tokens = tokens + [sep_token]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        label_ids = [label_map[label] for label in labels]
        label_ids = label_ids + [label_map["O"]] * (max_seq_length - len(label_ids))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        selection_matrix = np.zeros((max_seq_length, len(relation_list), max_seq_length), dtype=np.int)
        none = relation_map["NA"]
        selection_matrix[:, none, :] = 1
        for (s, p, o) in example.triples:

            if s > max_seq_length-1 or o > max_seq_length -1:
                continue

            selection_matrix[s, p, o] = 1
            selection_matrix[s, none, o] = 0

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels: %s" % " ".join([str(x) for x in labels]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("triples: %s" % " ".join([str(x) for x in example.triples]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          selection_matrix=selection_matrix))
    return features


processors = {
    "SpaceEval2015": MultiHeadSelectionDataProcessor
}
