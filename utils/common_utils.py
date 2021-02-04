import collections
import os
import sys
from itertools import groupby


def mark_predict_error(in_path):
    parent_dir, _ = os.path.split(in_path)
    out_path = os.path.join(parent_dir, "predict_error.txt")

    with open(in_path, 'r', encoding="utf-8") as f:
        lines = [line.strip().split() for line in f]
        for i, line in enumerate(lines):
            if not line:
                continue
            g_label, p_label = line[1:3]
            pre_p_label = lines[i - 1][2] if i > 0 and lines[i - 1] else 'O'

            if p_label.startswith("I-"):
                if pre_p_label == 'O':
                    line.append("start with I- error")
                    continue
                elif pre_p_label.__contains__("-"):
                    _, _, pre_tag = pre_p_label.partition("-")
                    _, _, cur_tag = p_label.partition('-')
                    if pre_tag != cur_tag:
                        line.append("inconsistent I-")
                        continue
            if g_label != p_label:
                line.append(g_label + ' ' + "error")
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(" ".join(line) + '\n')


def _is_divider(line: str) -> bool:
    if line.strip() == '':
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


def _is_start_of_chunk(labels, index):
    if labels[index] == "O":
        return False
    if index == 0 or labels[index - 1] == "O":
        return True

    cur_tag, cur_label = labels[index].split("-")
    pre_tag, pre_label = labels[index - 1].split("-")

    if cur_tag == "B" or cur_label != pre_label:
        return True
    return False


def _is_end_of_chunk(labels, index):
    if labels[index] == "O":
        return False
    if index == len(labels) - 1 or labels[index + 1] == "O":
        return True

    cur_tag, cur_label = labels[index].split("-")
    next_tag, next_label = labels[index + 1].split("-")

    if next_tag == "B" or cur_label != next_label:
        return True
    return False


def eval_srl_v2(lines):
    strict_correct = 0
    strict_gold = 0
    total_predict_num = 0
    total_correct_num = 0
    total_gold_num = 0

    for is_divider, group_lines in groupby(lines, _is_divider):
        if not is_divider:
            gold_dict = collections.defaultdict(set)
            predict_dict = collections.defaultdict(set)
            correct = True
            tuples = [line.strip().split()[-2:] for line in group_lines]
            golds, predicts = zip(*tuples)
            gold_start, predict_start, gold_end, predict_end = -1, -1, -1, -1
            for i, (_, _) in enumerate(zip(golds, predicts)):
                if _is_start_of_chunk(golds, i):
                    gold_start = i
                if _is_end_of_chunk(golds, i):
                    tag, label = golds[gold_start].split("-")
                    if tag != "B":
                        continue
                    gold_dict[label].add((gold_start, i + 1))

                if _is_start_of_chunk(predicts, i):
                    predict_start = i

                if _is_end_of_chunk(predicts, i):
                    tag, label = predicts[predict_start].split("-")
                    if tag != "B":
                        continue
                    predict_dict[label].add((predict_start, i + 1))

            predict_num = 1
            correct_num = 1
            gold_num = 1

            if gold_dict.__contains__("mover"):
                if predict_dict.__contains__("trajector"):
                    predict_dict.pop("trajector")
                if predict_dict.__contains__("landmark"):
                    predict_dict.pop("landmark")
            if gold_dict.__contains__("trajector") or gold_dict.__contains__("landmark"):
                if predict_dict.__contains__("mover"):
                    predict_dict.pop("mover")

            for key in gold_dict.keys():
                gold_set = gold_dict.get(key)
                pred_set = predict_dict.get(key) if predict_dict.__contains__(key) else set()
                correct_set = gold_set & pred_set
                gold_num *= len(gold_set)
                correct_num *= len(correct_set)
                if gold_set != pred_set:
                    correct = False
            if predict_dict.keys() != gold_dict.keys():
                correct = False
                correct_num = 0

            for k, v in predict_dict.items():
                predict_num *= len(v)

            total_predict_num += predict_num
            total_correct_num += correct_num
            total_gold_num += gold_num

            if correct:
                strict_correct += 1
            strict_gold += 1

    accuracy = strict_correct / strict_gold
    precision = total_correct_num / total_predict_num
    recall = total_correct_num / total_gold_num
    f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return accuracy, total_correct_num, total_predict_num, total_gold_num, precision, recall, f_1


def eval_srl(lines):
    strict_correct = 0
    strict_gold = 0
    total_predict_num = 0
    total_correct_num = 0
    total_gold_num = 0

    for is_divider, group_lines in groupby(lines, _is_divider):
        if not is_divider:
            gold_dict = collections.defaultdict(set)
            predict_dict = collections.defaultdict(set)
            correct = True
            tuples = [line.strip().split()[-2:] for line in group_lines]
            # for _, gold, predict in tuples:
            #     if gold != predict:
            #         correct = False
            #         break
            # strict_gold += 1
            # if correct:
            #     strict_correct += 1
            # else:
            #     strict_correct += 0
            golds, predicts = zip(*tuples)
            gold_start, predict_start, gold_end, predict_end = -1, -1, -1, -1
            for i, (_, _) in enumerate(zip(golds, predicts)):
                if _is_start_of_chunk(golds, i):
                    gold_start = i
                if _is_end_of_chunk(golds, i):
                    tag, label = golds[gold_start].split("-")
                    if tag != "B":
                        continue
                    gold_dict[label].add((gold_start, i + 1))

                if _is_start_of_chunk(predicts, i):
                    predict_start = i

                if _is_end_of_chunk(predicts, i):
                    tag, label = predicts[predict_start].split("-")
                    if tag != "B":
                        continue
                    predict_dict[label].add((predict_start, i + 1))

            predict_num = 1
            correct_num = 1
            gold_num = 1

            # if gold_dict.__contains__("mover"):
            #     if predict_dict.__contains__("trajector"):
            #         predict_dict.pop("trajector")
            #     if predict_dict.__contains__("landmark"):
            #         predict_dict.pop("landmark")
            # if gold_dict.__contains__("trajector") or gold_dict.__contains__("landmark"):
            #     if predict_dict.__contains__("mover"):
            #         predict_dict.pop("mover")

            for key in gold_dict.keys():
                gold_set = gold_dict.get(key)
                pred_set = predict_dict.get(key) if predict_dict.__contains__(key) else set()
                correct_set = gold_set & pred_set
                gold_num *= len(gold_set)
                correct_num *= len(correct_set)
                if gold_set != pred_set:
                    correct = False

            if predict_dict.keys() != gold_dict.keys():
                correct = False
                correct_num = 0

            for k, v in predict_dict.items():
                predict_num *= len(v)

            total_predict_num += predict_num
            total_correct_num += correct_num
            total_gold_num += gold_num

            if correct:
                strict_correct += 1
            strict_gold += 1

    accuracy = strict_correct / strict_gold if strict_gold != 0 else 0
    precision = total_correct_num / total_predict_num if total_predict_num != 0 else 0
    recall = total_correct_num / total_gold_num if total_gold_num != 0 else 0
    f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return strict_gold, accuracy, total_correct_num, total_predict_num, total_gold_num, precision, recall, f_1


def to_srl_format(in_path):
    parent_dir, _ = os.path.split(in_path)
    out_path = os.path.join(parent_dir, "predict_srl_err.txt")
    out_lines = []
    with open(in_path, 'r', encoding="utf-8") as f:
        for is_divider, group_lines in groupby(f, _is_divider):
            if not is_divider:
                gold_dict = collections.defaultdict(list)
                predict_dict = collections.defaultdict(list)
                correct = True
                tuples = [line.strip().split() for line in group_lines]
                for _, _, gold, predict in tuples:
                    if gold != predict:
                        correct = False
                        break
                # if correct:
                #     continue
                tokens, _, golds, predicts = zip(*tuples)
                sentence = " ".join(tokens)
                gold_start, predict_start, gold_end, predict_end = -1, -1, -1, -1
                for i, (_, _) in enumerate(zip(golds, predicts)):
                    if _is_start_of_chunk(golds, i):
                        gold_start = i
                    if _is_end_of_chunk(golds, i):
                        tag, label = golds[gold_start].split("-")
                        gold_dict[label].append("".join(tokens[gold_start:i + 1]) + "(" + tag + ")")

                    if _is_start_of_chunk(predicts, i):
                        predict_start = i

                    if _is_end_of_chunk(predicts, i):
                        tag, label = predicts[predict_start].split("-")
                        predict_dict[label].append("".join(tokens[predict_start:i + 1]) + "(" + tag + ")")

                out_lines.append(sentence)
                out_lines.append(str(correct))

                for key in set(list(gold_dict.keys()) + list(predict_dict.keys())):
                    gold_list = "[]" if not gold_dict.__contains__(key) else str(gold_dict[key])
                    predict_list = "[]" if not predict_dict.__contains__(key) else str(predict_dict[key])
                    line = "{}\tgold:{}\tpredict:{}".format(key, gold_list, predict_list)
                    out_lines.append(line)
                out_lines.append("")
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in out_lines:
            f.write(line + '\n')


def post_process_todlink(in_path):
    parent_dir, _ = os.path.split(in_path)
    out_path = os.path.join(parent_dir, "predict_TODlink_revised.txt")
    output = []
    with open(in_path, "r", encoding="utf-8") as in_file:
        for is_divider, group_lines in groupby(in_file, _is_divider):
            if not is_divider:
                tuples = [line.strip().split() for line in group_lines]
                tokens, tags, gold_labels, predict_labels = (list(t) for t in zip(*tuples))
                tag_start, tag_end, label_start, label_end = -1, -1, -1, -1
                current_label = ""
                spatial_entities = []
                mover_start, mover_end = -1, -1

                for i, (tag, p_label) in enumerate(zip(tags, predict_labels)):
                    label_type = p_label.split("-")[-1]
                    tag_type = tag.split("-")[-1]

                    if _is_start_of_chunk(tags, i):
                        tag_start = i
                    if _is_end_of_chunk(tags, i):
                        tag_end = i
                        # if tag_type == "SpatialEntity":
                        #     spatial_entities.append((tag_start, tag_end))

                    if _is_start_of_chunk(predict_labels, i):
                        label_start = i
                    if _is_end_of_chunk(predict_labels, i):
                        current_label = label_type
                        label_end = i
                        # if label_type == "mover":
                        #     mover_start, mover_end = label_start, label_end

                    if _is_end_of_chunk(predict_labels, i) and label_start < tag_start:
                        print("ERROR")
                        # exit(-1)

                    if _is_end_of_chunk(tags, i):
                        if tag_end != label_end or tag_start != label_start:
                            if "B-" + current_label in predict_labels[tag_start:tag_end + 1] or \
                                    "I-" + current_label in predict_labels[tag_start:tag_end + 1]:
                                for j in range(tag_start, tag_end + 1):
                                    predict_labels[j] = "B-" + current_label if j == tag_start else "I-" + current_label
                output.extend([" ".join(_tuple) for _tuple in zip(tokens, tags, gold_labels, predict_labels)])
                output.append("")
    print(eval_srl(output))
    with open(out_path, "w", encoding="utf-8") as out_file:
        for line in output:
            out_file.write(line + "\n")


def post_process_mlink(in_path, mode=None):
    parent_dir, _ = os.path.split(in_path)
    out_path = os.path.join(parent_dir, "predict_MLINK_revised.txt")
    output = []
    with open(in_path, "r", encoding="utf-8") as in_file:
        for is_divider, group_lines in groupby(in_file, _is_divider):
            if not is_divider:
                tuples = [line.strip().split() for line in group_lines]
                tokens, tags, gold_labels, predict_labels = (list(t) for t in zip(*tuples))
                tag_start, tag_end, label_start, label_end = -1, -1, -1, -1
                current_label = ""
                spatial_entities = []
                mover_start, mover_end = -1, -1

                for i, (tag, p_label) in enumerate(zip(tags, predict_labels)):
                    label_type = p_label.split("-")[-1]
                    tag_type = tag.split("-")[-1]

                    if _is_start_of_chunk(tags, i):
                        tag_start = i
                    if _is_end_of_chunk(tags, i):
                        tag_end = i
                        if tag_type == "SpatialEntity":
                            spatial_entities.append((tag_start, tag_end))

                    if _is_start_of_chunk(predict_labels, i):
                        label_start = i
                    if _is_end_of_chunk(predict_labels, i):
                        current_label = label_type
                        label_end = i
                        if label_type == "mover":
                            mover_start, mover_end = label_start, label_end

                    if _is_end_of_chunk(predict_labels, i) and label_start < tag_start:
                        print("ERROR")
                        exit(-1)

                    if _is_end_of_chunk(tags, i):
                        if (tag_end != label_end or tag_start != label_start) and current_label != "O":
                            if "B-mover" in predict_labels[tag_start:tag_end + 1]:
                                for j in range(tag_start, tag_end + 1):
                                    predict_labels[j] = "B-" + current_label if j == tag_start else "I-" + current_label

                if mode == "greed":
                    if "B-mover" not in predict_labels and len(spatial_entities) != 0:
                        index = 0
                        min_distance = 100000
                        for i, (start, end) in enumerate(spatial_entities):
                            distance = min(abs(start - mover_end), abs(end - mover_start))
                            if distance < min_distance:
                                index = i
                        begin, end = spatial_entities[index]
                        for i in range(begin, end + 1):
                            predict_labels[i] = "B-mover" if i == begin else "I-mover"

                output.extend([" ".join(_tuple) for _tuple in zip(tokens, tags, gold_labels, predict_labels)])
                output.append("")
    print(eval_srl(output))
    with open(out_path, "w", encoding="utf-8") as out_file:
        for line in output:
            out_file.write(line + "\n")


def process_no_trigger(in_path, tags_path):
    parent_dir, _ = os.path.split(in_path)
    out_path = os.path.join(parent_dir, "no_trigger_revised.txt")
    output = []
    line_tags = []
    with open(tags_path, "r", encoding="utf-8") as in_file:
        line_tags = [line.split("\t")[2].split(" ") if line.strip() != "" else [] for line in in_file]
    idx = 0
    with open(in_path, "r", encoding="utf-8") as in_file:
        for is_divider, group_lines in groupby(in_file, _is_divider):
            if not is_divider:
                tuples = [line.strip().split() for line in group_lines]
                tokens, ori_tags, gold_labels, predict_labels = (list(t) for t in zip(*tuples))
                tags = line_tags[idx]
                idx += 1
                tag_start, tag_end, label_start, label_end = -1, -1, -1, -1

                for i, (tag, p_label) in enumerate(zip(tags, predict_labels)):
                    if _is_start_of_chunk(tags, i):
                        tag_start = i
                    if _is_end_of_chunk(tags, i):
                        tag_end = i
                    if _is_start_of_chunk(predict_labels, i):
                        label_start = i
                    if _is_end_of_chunk(predict_labels, i):
                        label_end = i
                    if _is_end_of_chunk(predict_labels, i):
                        if tag_end != label_end or tag_start != label_start:
                            for j in range(label_start, label_end + 1):
                                predict_labels[j] = "O"

                if "B-landmark" not in predict_labels and "B-trajector" not in predict_labels:
                    trajector_index = -1
                    for i, tag in enumerate(tags):
                        if tag == "B-Event" and gold_labels[i] == "B-trajector":
                            predict_labels[i] = "B-trajector"
                            trajector_index = i
                        if tag == "I-Event" and gold_labels[i] == "I-trajector":
                            predict_labels[i] = "I-trajector"
                    if trajector_index >= 0:
                        for i in range(trajector_index, -1, -1):
                            if tags[i] == "I-Place":
                                predict_labels[i] = "I-landmark"
                            if tags[i] == "B-Place":
                                predict_labels[i] = "B-landmark"
                                break
                output.extend([" ".join(_tuple) for _tuple in zip(tokens, ori_tags, gold_labels, predict_labels)])
                output.append("")
    print(eval_srl(output))
    with open(out_path, "w", encoding="utf-8") as out_file:
        for line in output:
            out_file.write(line + "\n")


if __name__ == "__main__":
    # to_srl_format("D:\项目\军事问答\project\pytorch_nlp\output\srl_output\predict.txt")
    # f = open("D:\项目\军事问答\project\pytorch_nlp\predict\predict.txt", "r", encoding="utf-8")
    # eval_srl(f.readlines())
    # process_no_trigger(
    #     r"D:\项目\军事问答\project\relationwithrule\data\BIO\all+reason_seed_7_new\transformer_crf_head_6\withoutconj.txt", \
    #     r"D:\项目\军事问答\project\relationwithrule\data\BIO\all+reason_seed_7_new\transformer_crf_head_6\no_trigger.txt")
    #
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7/lstm_crf/mlink.txt")
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7/lstm_crf/mlink.txt", "greed")
    # post_process_todlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7/lstm_crf/todlink.txt")
    #
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/lstm_crf/mlink.txt")
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/lstm_crf/mlink.txt", "greed")
    # post_process_todlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/lstm_crf/todlink.txt")
    #
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/transformer_crf/mlink.txt")
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/transformer_crf/mlink.txt", "greed")
    # post_process_todlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/transformer_crf/todlink.txt")
    #
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/transformer_crf_head_6/mlink.txt")
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/transformer_crf_head_6/mlink.txt",
    #                    "greed")
    # post_process_todlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all_seed_0/transformer_crf_head_6/todlink.txt")
    #
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7/transformer_crf_head_6/mlink.txt")
    # post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7/transformer_crf_head_6/mlink.txt",
    #                    "greed")
    # post_process_todlink(
    #     "D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7/transformer_crf_head_6/todlink.txt")


    post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7_new/transformer_crf_head_6/mlink.txt")
    post_process_mlink("D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7_new/transformer_crf_head_6/mlink.txt",
                       "greed")
    post_process_todlink(
        "D:/项目/军事问答/project/pytorch_nlp/result/12-05/all+reason_seed_7_new/transformer_crf_head_6/todlink.txt")

# mark_predict_error("D:\项目\军事问答\project\pytorch_nlp\output\conll_output\crf_span\predict.txt")
