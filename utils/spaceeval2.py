import sys
from itertools import groupby
from collections import defaultdict


def load_links(filepath):
    type2links = defaultdict(lambda :defaultdict(set))
    with open(filepath, "r", encoding="utf-8") as file:
        for _, groups in groupby(file, lambda x: x.strip() == ""):
            groups = list(groups)
            filename = groups[0].strip()
            for link in groups[1:]:
                link_type = link[:link.index("(")]
                type2links[filename][link_type].add(f"{filename}\t{link}")
    return type2links


def cal_prf(correct_num, pred_num, gold_num):
    pre = 0 if pred_num == 0 else correct_num / pred_num
    rec = 0 if gold_num == 0 else correct_num / gold_num

    if pre + rec == 0:
        f1 = 0
    else:
        f1 = 2 * pre * rec / (pre + rec)

    return pre, rec, f1


def evaluate(gold_path, pred_path):
    gold_result = load_links(gold_path)
    pred_result = load_links(pred_path)

    total_correct_num, total_pre_num, total_gold_num = 0, 0, 0

    for file in gold_result.keys():
        for link_type in gold_result[file].keys():
            gold_links = gold_result[file][link_type]
            pred_links = pred_result[file][link_type]
            correct_links = pred_links & gold_links

            gold_num = len(gold_links)
            pred_num = len(pred_links)
            correct_num = len(correct_links)
            if link_type == "MOVELINK" and file == "RideForClimateUSA.xml":
                print(file, correct_num)

            pre, rec, f1 = cal_prf(correct_num, pred_num, gold_num)

            print(f"{link_type: <10}\tPre:{pre:.4f}\tRec:{rec:.4f}\tF-1:{f1:.4f}")

            total_pre_num += pred_num
            total_gold_num += gold_num
            total_correct_num += correct_num

    pre, rec, f1 = cal_prf(total_correct_num, total_pre_num, total_gold_num)
    print(f"{'OVERALL': <10}\tPre:{pre:.4f}\tRec:{rec:.4f}\tF-1:{f1:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python spaceeval.py /path/to/gold/file  /path/to/result/file")
    else:
        g_path, p_path = sys.argv[1:]
        evaluate(g_path, p_path)
