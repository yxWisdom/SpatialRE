import logging
import os
from itertools import islice, groupby

from utils.common_utils import eval_srl, to_srl_format
from utils.seq_labeling_eval import classification_report

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    return len(line) == 0


def _get_padded_labels(tokens, label, use_x_tag=False):
    padding_label = label.replace('B-', 'I-')
    return [label] + ["X" if use_x_tag and token.startswith("##") else padding_label for token in tokens[1:]]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, predicates=None, tags=None, tokens=None, labels=None):
        self.guid = guid
        self.text = text
        self.tokens = tokens
        self.labels = labels
        self.predicates = predicates
        self.tags = tags


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, predicate_mask, tag_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.predicate_mask = predicate_mask
        self.tag_ids = tag_ids


class BertSRLDataProcessor:
    def __init__(self, data_dir, use_x_tag=True, keep_span=False):
        self.labels = set()
        self.use_x_tag = use_x_tag
        self.keep_span = keep_span

        self.train_examples = self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")
        self.dev_examples = self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")
        self.test_examples = None

        tmp_labels = set()
        tmp_tags = set()

        for example in self.train_examples:
            for tag, label in zip(example.tags, example.labels):
                tmp_labels.add(label)
                tmp_tags.add(tag)

        if use_x_tag:
            tmp_labels.add("X")
            tmp_tags.add("X")

        self.labels = sorted(tmp_labels)
        self.tags = sorted(tmp_tags)

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
            _, tokens_str, tags_str, labels_str = line.split('\t')
            tokens = tokens_str.split()
            tags = tags_str.split()
            labels = labels_str.split()
            text = "".join(tokens)
            assert len(tokens) == len(labels) == len(tags)
            examples.append(InputExample(guid=guid, text=text, tokens=tokens, tags=tags, labels=labels))
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

    def decode(self, predict_label_ids, tokenizer, set_type="dev"):
        examples = self.dev_examples if set_type == 'dev' else self.test_examples
        lines = []

        for example, p_label_ids in zip(examples, predict_label_ids):
            token_pieces_list = [tokenizer.tokenize(token) for token in example.tokens]
            token_piece_lens = [len(piece) for piece in token_pieces_list]

            it = iter(self.labels[label_id] for label_id in p_label_ids[1:])
            p_label_pieces_list = [list(islice(it, 0, i)) for i in token_piece_lens]

            for token, tag, g_label, token_pieces, p_label_pieces in zip(example.tokens, example.tags,example.labels,
                                                                    token_pieces_list, p_label_pieces_list):
                if self.keep_span:
                    lines.append((token, tag, g_label, p_label_pieces[0]))
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
            'srl_sentence'] = "total:{} accuracy:{:.4f}, gold:{},predict{},correct:{},precision:{:.4f}, recall{:.4f}, f1:{:.4f}". \
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


def convert_examples_to_features(examples, label_list, tag_list, max_seq_length,
                                 tokenizer, use_x_tag=True,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
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
    label_map.update({cls_token: label_map["O"], sep_token: label_map["O"]})

    label_reverse_map = {v: k for k, v in label_map.items()}

    tag_map = {tag: i for i, tag in enumerate(tag_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens, labels, tags, predicate_mask = [], [], [], [0] * max_seq_length

        for token, label, tag in zip(example.tokens, example.labels, example.tags):
            sub_tokens = tokenizer.tokenize(token)
            tokens.extend(sub_tokens)
            labels.extend(_get_padded_labels(sub_tokens, label, use_x_tag))
            tags.extend(_get_padded_labels(sub_tokens, tag, use_x_tag))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            labels = labels[:(max_seq_length - special_tokens_count)]
            tags = tags[:(max_seq_length - special_tokens_count)]

        tokens = tokens + [sep_token]
        labels = labels + [sep_token]
        tags = tags + ["O"]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            labels += [sep_token]
            tags += ["O"]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            labels = labels + [cls_token]
            tags = tags + ["O"]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            labels = [cls_token] + labels
            tags = ["O"] + tags
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_map[label] if label_map.__contains__(label) else -1 for label in labels]
        tag_ids = [tag_map[tag] for tag in tags]

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            label_ids = ([-1] * padding_length) + label_ids
            tag_ids = ([tag_map["O"]] * padding_length) + tag_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            label_ids = label_ids + ([-1] * padding_length)
            tag_ids = tag_ids + ([tag_map["O"]] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        predicate_mask = [1 if label_id >= 0 and label_reverse_map[label_id].endswith("trigger") else 0 for label_id in
                          label_ids]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(predicate_mask) == max_seq_length
        assert len(tag_ids) == max_seq_length

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
            logger.info("predict_mask: %s" % " ".join([str(x) for x in predicate_mask]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          tag_ids=tag_ids,
                          predicate_mask=predicate_mask))
    return features


processors = {
    "ner_x_span": {
        "name": BertSRLDataProcessor,
        "params": {
            "use_x_tag": True,
            "keep_span": True
        },
    },
    "ner_x": {
        "name": BertSRLDataProcessor,
        "params": {
            "use_x_tag": True,
            "keep_span": False
        },
    },
    "ner_span": {
        "name": BertSRLDataProcessor,
        "params": {
            "use_x_tag": False,
            "keep_span": True
        },
    },
    "ner": {
        "name": BertSRLDataProcessor,
        "params": {
            "use_x_tag": False,
            "keep_span": False
        }
    }
}
