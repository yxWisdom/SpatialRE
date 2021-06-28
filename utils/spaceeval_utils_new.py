# -*- coding: utf-8 -*-
"""
@Date ： 2021/1/11 19:27
@Author ： xyu
"""
from collections import defaultdict
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from utils.spaceeval_utils import Metrics

# eval_optional_roles是否评测move links的可选角色
# eval_null_roles是否允许角色是空的

qs_roles = ["trigger", "trajector", "landmark"]
move_core_roles = ["trigger", "mover"]
move_optional_roles = ["source", "goal", "landmark", "midPoint", "pathID", "motion_signalID"]

multi_roles = ["midPoint", "pathID", "motion_signalID"]

link2roles = {
    "QSLINK": ["trigger", "trajector", "landmark"],
    "OLINK": ["trigger", "trajector", "landmark"],
    "MOVELINK": ["trigger", "mover", "source", "goal", "ground", "midPoint", "pathID", "motion_signalID"]
}


def parse_no_trigger(element: Element, element_dict: dict, metric: Metrics):
    if metric == Metrics.OFFICIAL:
        return []
    trigger = element.attrib.get("trigger", "")
    trajector = element.attrib.get("trajector", "")
    landmark = element.attrib.get("landmark", "")
    if trajector in element_dict and landmark in element_dict and trigger not in element_dict:
        return [("NoTrigger", "", trajector, landmark)]
    return []


def parse_qs_o_link(element: Element, element_dict: dict, metric: Metrics):
    trigger = element.attrib.get("trigger", "")
    trajector = element.attrib.get("trajector", "")
    landmark = element.attrib.get("landmark", "")

    has_trigger = trigger in element_dict
    has_two_role = trajector in element_dict and landmark in element_dict

    if metric == Metrics.OFFICIAL and (not has_trigger or not has_two_role):
        return []

    if metric == Metrics.STRICT and (not has_two_role):
        return []

    link_tuple = (element.tag,)
    for role in [trigger, trajector, landmark]:
        if role not in element_dict:
            role = ""
        link_tuple += (role,)

    return [link_tuple]


def parse_move_link(element: Element, element_dict: dict, metric: Metrics, eval_optional_roles=False,
                    eval_null_mover=False):
    trigger = element.attrib.get("trigger", "")
    mover = element.attrib.get("mover", "")

    if trigger not in element_dict:
        trigger = ""
    if mover not in element_dict:
        mover = ""

    tuple_ = (element.tag, trigger, mover)

    if not eval_null_mover and mover not in element_dict:
        return []

    if metric == Metrics.OFFICIAL:
        return [] if trigger not in element_dict else [tuple_]

    if metric == Metrics.STRICT:
        if trigger not in element_dict:
            return []

    if eval_optional_roles:
        if element.attrib.get("adjunctID", "") != "":
            element.attrib["motion_signalID"] = element.attrib.get("adjunctID")
        for role in move_optional_roles:
            if role in multi_roles:
                role_ids = element.attrib.get(role, "").replace(" ", "").split(",")
                role_ids = filter(lambda x: x in element_dict, role_ids)
                role_ids = ",".join(sorted(role_ids))
                tuple_ += (role_ids,)
            else:
                role_id = element.attrib.get(role, "")
                if role_id.__contains__(","):
                    print(role, "ERROR")
                    exit(1)
                if role_id not in element_dict:
                    role_id = ""
                tuple_ += (role_id,)
    return [tuple_]


def load_links_from_file(path, metric: Metrics, eval_optional_roles=True, eval_null_mover=False):
    tree = ElementTree.parse(path)
    tags = tree.getroot().find("TAGS")

    link2list = defaultdict(set)
    element_dict = {}

    for element in tags:
        if 'start' in element.attrib and int(element.attrib['start']) != -1 and element.attrib['id'] != "":
            element_dict[element.attrib['id']] = int(element.attrib['start'])

    # if "" in element_dict:
    #     exit(1)

    for element in tags:
        if element.tag == "QSLINK" or element.tag == "OLINK":
            if element.attrib['trajector'] == "pl0":
                print()

            link2list[element.tag].update(parse_qs_o_link(element, element_dict, metric))
            link2list["NoTrigger"].update(parse_no_trigger(element, element_dict, metric))
        elif element.tag == "MOVELINK":
            link2list[element.tag].update(parse_move_link(element, element_dict, metric, eval_optional_roles,
                                                          eval_null_mover))
    return link2list


def save(input_dir, output_path, eval_optional_roles):
    metric = Metrics.STRICT if eval_optional_roles else Metrics.OFFICIAL

    with open(output_path, "w", encoding="utf-8") as f:
        for root, _, files in os.walk(input_dir):
            for file in files:
                f.write(file + "\n")
                path = os.path.join(root, file)
                link_dict = load_links_from_file(path, metric=metric, eval_optional_roles=eval_optional_roles,
                                                 eval_null_mover=False)

                for link_type, role_types in link2roles.items():
                    links = link_dict.get(link_type, [])
                    links = sorted(list(links))
                    for link in links:
                        role2value_list = [f"{role_type}={role_value}" for role_type, role_value in
                                           zip(role_types, link[1:])]
                        f.write(f"{link_type}({', '.join(role2value_list)})\n")
                f.write("\n")


def print_links(file_dir):
    all_link_dict = defaultdict(int)
    for root, _, files in os.walk(file_dir):
        for file in files:
            print()
            print(file)
            path = os.path.join(root, file)
            link_dict = load_links_from_file(path, metric=Metrics.STRICT, eval_optional_roles=True,
                                             eval_null_mover=False)
            for link_type, links in link_dict.items():
                # print(link_type, len(links))
                if link_type == "MOVELINK":
                    continue
                print(link_type)
                # links.sort()
                for link in links:
                    print(link)
                all_link_dict[link_type] += len(links)
    print(all_link_dict)


if __name__ == '__main__':
    import os

    # file_path = "D:\\项目\\军事问答\\project\\pytorch_nlp\\dataset\\MHS\\sample_5\\xml\\Tikal.xml"
    # load_links_from_file(file_path, metric=Metrics.STRICT, eval_optional_roles=True, eval_null_mover=True)

    # # data_dir = "D:\\项目\空间关系识别\\repo\\spatialie\\data\\SpaceEval2015\\raw_data\\training++"
    # data_dir = "D:\\项目\\空间关系识别\\repo\\spatialie\\data\\SpaceEval2015\\predict_result\\MHS\\configuration3\\part\\XML"

    gold_dir = "D:\\项目\\军事问答\\project\\pytorch_nlp\\dataset\\MHS\\sample_5\\xml"
    gold_part_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\main roles(Table 3)\\gold.txt"
    gold_full_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\entire relations(Table 4)\\gold.txt"
    save(gold_dir, gold_part_path, eval_optional_roles=False)
    save(gold_dir, gold_full_path, eval_optional_roles=True)

    sieve_dir = "D:\\项目\\军事问答\\project\\SpatialRelEx\\src\\output"
    sieve_part_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\main roles(Table 3)\\sieve.txt"
    sieve_full_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\entire relations(Table 4)\\sieve.txt"
    save(sieve_dir, sieve_part_path, eval_optional_roles=False)
    save(sieve_dir, sieve_full_path, eval_optional_roles=True)

    seq_tag_dir = "D:\\项目\\空间关系识别\\repo\\spatialie\\data\\SpaceEval2015\\predict_result\\SpRL\\configuration3\\full\\XML"
    seq_tag_part_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\main roles(Table 3)\\seqTag.txt"
    seq_tag_full_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\entire relations(Table 4)\\seqTag.txt"
    save(seq_tag_dir, seq_tag_part_path, eval_optional_roles=False)
    save(seq_tag_dir, seq_tag_full_path, eval_optional_roles=True)

    multi_roles = ["motion_signalID"]
    ours_dir = "D:\\项目\\空间关系识别\\repo\\spatialie\\data\\SpaceEval2015\\predict_result\\MHS\\configuration3\\full\\XML"
    ours_part_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\main roles(Table 3)\\di4sr.txt"
    ours_full_path = "D:\\项目\\投稿\\IJCAI2021\\attachment\\entire relations(Table 4)\\di4sr.txt"
    save(ours_dir, ours_part_path, eval_optional_roles=False)
    save(ours_dir, ours_full_path, eval_optional_roles=True)
