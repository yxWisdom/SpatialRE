import collections
import logging
import os
from itertools import islice, groupby, chain

from utils.common_utils import eval_srl, to_srl_format
from utils.seq_labeling_eval import classification_report

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    return len(line.strip()) == 0


def _get_padded_labels(tokens, label, use_x_tag=False):
    padding_label = label.replace('B-', 'I-')
    return [label] + ["X" if use_x_tag and token.startswith("##") else padding_label for token in tokens[1:]]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, predicates=None, tags=None, tokens=None, pretags=None, labels=None):
        self.guid = guid
        self.text = text
        self.tokens = tokens
        self.labels = labels
        self.predicates = predicates
        self.tags = tags
        self.pretags = pretags


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, predicate_mask, tag_ids, pretag_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.predicate_mask = predicate_mask
        self.tag_ids = tag_ids
        self.pretag_ids = pretag_ids


class BertSRLWithRuleDataProcessor:
    def __init__(self, data_dir, use_x_tag=False, keep_span=True):
        self.labels = set()
        self.use_x_tag = use_x_tag
        self.keep_span = keep_span

        self.train_examples = self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")
        self.dev_examples = self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")
        self.test_examples = None
        self._read_labels()
        # tmp_labels = set()
        # tmp_tags = set()
        #
        # for example in self.train_examples:
        #     for tag, label in zip(example.tags, example.labels):
        #         tmp_labels.add(label)
        #         tmp_tags.add(tag)
        #
        # if use_x_tag:
        #     tmp_labels.add("X")
        #     tmp_tags.add("X")
        #
        # self.labels = sorted(tmp_labels)
        # self.tags = sorted(tmp_tags)

    def _read_labels(self):
        label_set = set(chain.from_iterable(map(lambda x: x.labels, self.train_examples)))
        tag_set = set(chain.from_iterable(map(lambda x: x.tags, self.train_examples)))
        if self.use_x_tag:
            label_set.add("X")
            tag_set.add("X")
        self.labels = sorted(sorted(label_set), key=lambda x: len(x))
        self.tags = sorted(sorted(tag_set), key=lambda x: len(x))

    @classmethod
    def _read_data(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if len(line.strip()) > 0]
            return lines

    @classmethod
    def _create_example(cls, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            _, tokens_str, tags_str, pretags_str, labels_str = line.split('\t')
            tokens = tokens_str.split()
            tags = tags_str.split()
            pretags = pretags_str.split()
            labels = labels_str.split()
            text = " ".join(tokens)
            assert len(tokens) == len(labels) == len(tags)
            examples.append(
                InputExample(guid=guid, text=text, tokens=tokens, tags=tags, labels=labels, pretags=pretags))
        return examples

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return self.labels

    def get_tags(self):
        return self.tags

    def get_feature_size(self):
        return [2, len(self.tags), len(self.labels)]

    def decode(self, predict_label_ids, tokenizer, set_type="dev"):
        examples = self.dev_examples if set_type == 'dev' else self.test_examples
        lines = []

        for example, p_label_ids in zip(examples, predict_label_ids):
            token_pieces_list = [tokenizer.tokenize(token) for token in example.tokens]
            token_piece_lens = [len(piece) for piece in token_pieces_list]

            it = iter(self.labels[label_id] for label_id in p_label_ids)
            p_label_pieces_list = [list(islice(it, 0, i)) for i in token_piece_lens]

            for token, tag, pretag, g_label, token_pieces, p_label_pieces in zip(example.tokens, example.tags,
                                                                                 example.pretags, example.labels,
                                                                                 token_pieces_list,
                                                                                 p_label_pieces_list):
                if self.keep_span:
                    try:
                        lines.append((token, tag, g_label, p_label_pieces[0]))
                    except IndexError:
                        lines.append((token, tag, g_label, "O"))
                else:
                    tokens, p_labels = [], []
                    for _token, _label in zip(token_pieces, p_label_pieces):
                        if _token.startswith('##'):
                            tokens[-1] += _token[2:]
                        else:
                            tokens.append(_token)
                            p_labels.append(_label)
                    g_labels = _get_padded_labels(token_pieces, g_label)
                    lines.extend(list(zip(tokens, tag, g_labels, p_labels)))
            lines.append([])
        return lines

    def evaluate(self, predict_label_ids, tokenizer, set_type="dev"):
        eval_dict = {}
        lines = self.decode(predict_label_ids, tokenizer, set_type)
        _, _, golds, preds = zip(*(filter(lambda x: len(x) > 0, lines)))
        eval_dict['srl_token'] = classification_report(y_true=list(golds), y_pred=list(preds), digits=4)
        lines = list(map(lambda x: ' '.join(x), lines))

        accuracy, total_correct_num, total_predict_num, total_gold_num, precision, recall, f_1 = eval_srl(lines)
        eval_dict[
            'srl_sentence'] = "total:{} accuracy:{:.4f}, correct:{},predict{},gold:{},precision:{:.4f}, recall{:.4f}, f1:{:.4f}". \
            format(len(self.dev_examples), accuracy, total_correct_num, total_predict_num, total_gold_num,
                   precision, recall, f_1)
        eval_dict['accuracy'] = accuracy
        eval_dict['f-1'] = f_1
        return eval_dict

    def save_predict(self, output_path, predict_label_ids, tokenizer, set_type="dev"):
        lines = self.decode(predict_label_ids, tokenizer, set_type)
        path = os.path.join(output_path, "predict.txt")
        with open(path, 'w', encoding='utf-8') as writer:
            for line in lines:
                writer.write(' '.join(line) + '\n')
        to_srl_format(path)

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
                                     mask_padding_with_zero=True):
        label_map = {label: i for i, label in enumerate(self.labels)}
        # label_map.update({cls_token: label_map["O"], sep_token: label_map["O"]})
        tag_map = {tag: i for i, tag in enumerate(self.tags)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 1000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens, labels, tags, pretags, predicate_mask = [], [], [], [], [0] * max_seq_length

            for token, label, tag, pretag in zip(example.tokens, example.labels, example.tags, example.pretags):
                sub_tokens = tokenizer.tokenize(token)
                tokens.extend(sub_tokens)
                labels.extend(_get_padded_labels(sub_tokens, label, self.use_x_tag))
                tags.extend(_get_padded_labels(sub_tokens, tag, self.use_x_tag))
                pretags.extend(_get_padded_labels(sub_tokens, pretag, self.use_x_tag))

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

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            label_ids = [label_map[label] if label_map.__contains__(label) else -1 for label in labels]
            label_ids = label_ids[:max_seq_length] + [pad_token_label_id] * (max_seq_length - len(label_ids))

            tag_ids = [tag_map[tag] for tag in tags]
            tag_ids = tag_ids[:max_seq_length] + [tag_map["O"]] * (max_seq_length - len(tag_ids))

            pretag_ids = [label_map[pretag] for pretag in pretags]
            pretag_ids = pretag_ids[:max_seq_length] + [label_map["O"]] * (max_seq_length - len(pretag_ids))

            predicate_mask = [1 if label.endswith("trigger") else 0 for label in labels]
            predicate_mask = predicate_mask[:max_seq_length] + [0] * (max_seq_length - len(predicate_mask))

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

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(predicate_mask) == max_seq_length
            assert len(tag_ids) == max_seq_length
            assert len(pretag_ids) == max_seq_length

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
                logger.info("tag_ids: %s" % " ".join([str(x) for x in tag_ids]))
                logger.info("pretag_ids: %s" % " ".join([str(x) for x in pretag_ids]))
                logger.info("predict_mask: %s" % " ".join([str(x) for x in predicate_mask]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              predicate_mask=predicate_mask,
                              tag_ids=tag_ids,
                              pretag_ids=pretag_ids))
        return features


#
# def convert_examples_to_features(examples, label_list, tag_list, max_seq_length,
#                                  tokenizer, use_x_tag=True,
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
#     # label_map.update({cls_token: label_map["O"], sep_token: label_map["O"]})
#     tag_map = {tag: i for i, tag in enumerate(tag_list)}
#
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 1000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         tokens, labels, tags, pretags, predicate_mask = [], [], [], [], [0] * max_seq_length
#
#         for token, label, tag, pretag in zip(example.tokens, example.labels, example.tags, example.pretags):
#             sub_tokens = tokenizer.tokenize(token)
#             tokens.extend(sub_tokens)
#             labels.extend(_get_padded_labels(sub_tokens, label, use_x_tag))
#             tags.extend(_get_padded_labels(sub_tokens, tag, use_x_tag))
#             pretags.extend(_get_padded_labels(sub_tokens, pretag, use_x_tag))
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
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         label_ids = [label_map[label] if label_map.__contains__(label) else -1 for label in labels]
#         label_ids = label_ids[:max_seq_length] + [pad_token_label_id] * (max_seq_length - len(label_ids))
#
#         tag_ids = [tag_map[tag] for tag in tags]
#         tag_ids = tag_ids[:max_seq_length] + [tag_map["O"]] * (max_seq_length - len(tag_ids))
#
#         pretag_ids = [label_map[pretag] for pretag in pretags]
#         pretag_ids = pretag_ids + [label_map["O"]] * (max_seq_length - len(pretag_ids))
#
#         predicate_mask = [1 if label.endswith("trigger") else 0 for label in labels]
#         predicate_mask = predicate_mask[:max_seq_length] + [0] * (max_seq_length - len(predicate_mask))
#
#         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#
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
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#         assert len(label_ids) == max_seq_length
#         assert len(predicate_mask) == max_seq_length
#         assert len(tag_ids) == max_seq_length
#         assert len(pretag_ids) == max_seq_length
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
#             logger.info("tag_ids: %s" % " ".join([str(x) for x in tag_ids]))
#             logger.info("pretag_ids: %s" % " ".join([str(x) for x in pretag_ids]))
#             logger.info("predict_mask: %s" % " ".join([str(x) for x in predicate_mask]))
#
#         features.append(
#             InputFeatures(input_ids=input_ids,
#                           input_mask=input_mask,
#                           segment_ids=segment_ids,
#                           label_ids=label_ids,
#                           tag_ids=tag_ids,
#                           predicate_mask=predicate_mask,
#                           pretag_ids=pretag_ids))
#     return features
#

processors = {
    "ner_x_span": {
        "name": BertSRLWithRuleDataProcessor,
        "params": {
            "use_x_tag": True,
            "keep_span": True
        },
    },
    "ner_x": {
        "name": BertSRLWithRuleDataProcessor,
        "params": {
            "use_x_tag": True,
            "keep_span": False
        },
    },
    "ner_span": {
        "name": BertSRLWithRuleDataProcessor,
        "params": {
            "use_x_tag": False,
            "keep_span": True
        },
    },
    "ner": {
        "name": BertSRLWithRuleDataProcessor,
        "params": {
            "use_x_tag": False,
            "keep_span": False
        }
    }
}


def preprocessing(ruled_path, ori_path, target_path):
    sentence_dict = {}
    sentence_dict_2 = {}
    with open(ruled_path, "r", encoding="utf-8") as f:
        # lines = []

        for is_divider, group_lines in groupby(f, _is_divider):
            if not is_divider:
                group_lines = list(group_lines)
                fields = [line.strip().split(" ") for line in group_lines]
                # lines.append(fields)
                for line in fields:
                    if line[-2].endswith("trigger"):
                        line[-1] = line[-2]
                tokens, tags, labels, pretags = zip(*fields)
                key = " ".join(tokens) + " " + " ".join(labels)
                sentence_dict[key] = " ".join(pretags)
                sentence_dict_2[key] = group_lines

    # file_name = os.path.basename(ori_path)
    # target_path = os.path.join(target_dir, file_name)
    ordered_rule_list = []
    with open(ori_path, "r", encoding="utf-8") as in_file, open(target_path, "w", encoding="utf-8") as out_file:
        for line in in_file.readlines():
            idx_str, tokens_str, tags_str, labels_str = line.strip().split('\t')
            key = tokens_str + " " + labels_str

            # if not sentence_dict.__contains__(key):
            #     print(1)
            pretags = sentence_dict[key]
            ordered_rule_list.extend(sentence_dict_2[key])
            ordered_rule_list.append("\n")

            new_line = idx_str + "\t" + tokens_str + "\t" + tags_str + "\t" + pretags + "\t" + labels_str + "\n"
            out_file.write(new_line)

    file_name = os.path.basename(ori_path)
    file_path = os.path.dirname(ruled_path)
    rule_path = os.path.join(file_path, "rule.txt")
    if file_name.startswith("test"):
        with open(rule_path, "w", encoding="utf-8") as file:
            for line in ordered_rule_list:
                file.write(line)


def naive_rule_post_process(learn_path, rule_path):
    learn_list = []

    out_dir = os.path.dirname(learn_path)
    out_path = os.path.join(out_dir, "predict_rule_2.txt")
    with open(learn_path, "r", encoding="utf-8") as f1:
        for is_divider, group_lines in groupby(f1, _is_divider):
            if not is_divider:
                learn_list.append([line.strip().split(" ") for line in group_lines])
    rule_list = []
    with open(rule_path, "r", encoding="utf-8") as f2:
        for is_divider, group_lines in groupby(f2, _is_divider):
            if not is_divider:
                rule_list.append([line.strip().split(" ") for line in group_lines])

    merge_list = []
    with open(out_path, "w", encoding="utf-8") as f3:
        for learn, rule in zip(learn_list, rule_list):
            status = False
            for line in rule:
                # if line[0] == "02-Jan-2010":
                #     print(1)
                if line[-1].endswith("trigger") and line[-2].endswith("trigger"):
                    status = True
                    break
            arr = rule if status else learn
            arr = list(map(lambda x: " ".join(x), arr))
            merge_list.extend(arr)
            merge_list.append("")

        for line in merge_list:
            f3.write(line + '\n')
    print(os.path.abspath(learn_path))
    eval_all_link(learn_path)

    print(os.path.abspath(rule_path))
    eval_all_link(rule_path)

    print(os.path.abspath(out_path))
    eval_all_link(out_path)

    print("")


def eval_all_link(path):
    # movelink_list = []
    # nonMoveLink_list = []
    # with open(path, "r", encoding="utf-8") as f1:
    #     for is_divider, group_lines in groupby(f1, _is_divider):
    #         if not is_divider:
    #             # example = [line.strip().split(" ") for line in group_lines]
    #             lines = [line.strip() for line in group_lines]
    #             link_type = 0
    #             # b = 0
    #             for line in lines:
    #                 arr = line.strip().split(" ")
    #                 if arr[-2].endswith("trigger") and arr[1].endswith("MOTION"):
    #                     link_type = 1
    #                     break
    #
    #                 # if arr[-2].endswith("trigger") and arr[1].endswith("SPATIAL_SIGNAL"):
    #                 #     b = 1
    #
    #             # if link_type == 1 and b == 1:
    #             #     print(1)
    #             if link_type == 0:
    #                 nonMoveLink_list.extend(lines)
    #                 nonMoveLink_list.append('')
    #             else:
    #                 movelink_list.extend(lines)
    #                 movelink_list.append('')
    # logger.info("nonMoveLink:" + str(eval_srl(nonMoveLink_list)))
    # logger.info("MoveLink:" + str(eval_srl(movelink_list)))

    links_dict = collections.defaultdict(list)

    with open(path, "r", encoding="utf-8") as f1:
        for is_divider, group_lines in groupby(f1, _is_divider):
            if not is_divider:
                lines = [line.strip() for line in group_lines]
                link_type = ""
                for line in lines:
                    arr = line.strip().split(" ")
                    if arr[-2].endswith("trigger"):
                        link_type = arr[-2]
                        break
                links_dict[link_type].extend(lines)
                links_dict[link_type].append("")
    for link_type, link_list in links_dict.items():
        total_num, acc, correct_num, predict_num, gold_num, p, r, f_1 = eval_srl(link_list)
        info = "{} total_num:{} acc:{:.4f} c_num:{}, p_num:{},g_num:{},p:{:.4f}, r:{:.4f}, f_1:{:.4f}"\
            .format(link_type, total_num, acc, correct_num, predict_num, gold_num, p, r, f_1)
        logger.info(info)


def checkDifferenceBetweenLearnAndRule(learn_path, rule_path):
    learn_list = []

    out_dir = os.path.dirname(learn_path)
    out_path = os.path.join(out_dir, "predict_rule_2.txt")
    with open(learn_path, "r", encoding="utf-8") as f1:
        for is_divider, group_lines in groupby(f1, _is_divider):
            if not is_divider:
                learn_list.append([line.strip().split(" ") for line in group_lines])
    rule_list = []
    with open(rule_path, "r", encoding="utf-8") as f2:
        for is_divider, group_lines in groupby(f2, _is_divider):
            if not is_divider:
                rule_list.append([line.strip().split(" ") for line in group_lines])

    common_count = 0
    learn_count = 0
    rule_count = 0
    false_count = 0

    for learn, rule in zip(learn_list, rule_list):
        learn_correct = True
        rule_correct = True
        for line in learn:
            if line[-1] != line[-2]:
                learn_correct = False
                break
        for line in rule:

            if line[-1] != line[-2]:
                rule_correct = False
                break
        if learn_correct and rule_correct:
            common_count += 1
        elif learn_correct:
            learn_count += 1
        elif rule_correct:
            tokens = [line[0] for line in learn]
            print(" ".join(tokens))
            for line in learn:
                print(line)
            print()
            rule_count += 1
        else:
            false_count += 1
    print(common_count, learn_count, rule_count, false_count)


if __name__ == '__main__':
    # checkDifferenceBetweenLearnAndRule("../dataset/SpaceEval2015/SRL/AllLink/predict.txt",
    #                                    "../dataset/SpaceEval2015/SRL/AllLink/rule.txt")
    #
    # checkDifferenceBetweenLearnAndRule("../dataset/SpaceEval2015/SRL/MoveLink/predict.txt",
    #                                    "../dataset/SpaceEval2015/SRL/MoveLink/rule.txt")
    #
    # checkDifferenceBetweenLearnAndRule("../dataset/SpaceEval2015/SRL/NonMoveLink/predict.txt",
    #                                    "../dataset/SpaceEval2015/SRL/NonMoveLink/rule.txt")

    checkDifferenceBetweenLearnAndRule("D:/垃圾场/predict.txt",
                                       "D:/垃圾场/rule.txt")
    #

    # eval_all_link('../dataset/SpaceEval2015/SRL/AllLink/predict_rule_1.txt')
    # eval_all_link('../dataset/SpaceEval2015/SRL/MoveLink/predict_rule_1.txt')
    # eval_all_link('../dataset/SpaceEval2015/SRL/NonMoveLink/predict_rule_1.txt')

    # naive_rule_post_process('../dataset/SpaceEval2015/SRL/AllLink/predict.txt',
    #                         '../dataset/SpaceEval2015/SRL/AllLink/rule.txt')
    #
    # naive_rule_post_process('../dataset/SpaceEval2015/SRL/MoveLink/predict.txt',
    #                         '../dataset/SpaceEval2015/SRL/MoveLink/rule.txt')
    #
    # naive_rule_post_process('../dataset/SpaceEval2015/SRL/NonMoveLink/predict.txt',
    #                         '../dataset/SpaceEval2015/SRL/NonMoveLink/rule.txt')

    # ruled_dir = "D:\ProjectsRepository\IntellijProjects\spatialie\output\SpaceEval2015\processed_data\SRL"
    # ori_dir = "D:\ProjectsRepository\IntellijProjects\spatialie\data\SpaceEval2015\processed_data\SRL"
    # target_dir = "D:\ProjectsRepository\IntellijProjects\spatialie\data\SpaceEval2015\processed_data\SRL_rule"
    #
    # for dir_name in ["AllLink", "MoveLink", "NonMoveLink"]:
    #     for file_type in ["train.txt", "dev.txt", "test.txt"]:
    #         ruled_path = os.path.join(ruled_dir, dir_name, file_type)
    #         ori_path = os.path.join(ori_dir, dir_name, file_type)
    #         target_path = os.path.join(target_dir, dir_name, file_type)
    #         if file_type == "dev.txt":
    #             ruled_path = os.path.join(ruled_dir, dir_name, "test.txt")
    #             ori_path = os.path.join(ori_dir, dir_name, "test.txt")
    #         preprocessing(ruled_path, ori_path, target_path)
