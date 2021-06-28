# -*- coding: utf-8 -*-
"""
@Date ： 2021/1/11 19:27
@Author ： xyu
"""
from collections import defaultdict
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

# eval_optional_roles是否评测move links的可选角色
# eval_null_roles是否允许角色是空的

from enum import Enum

qs_roles = ["trigger", "trajector", "landmark"]
move_core_roles = ["trigger", "mover"]
move_optional_roles = ["source", "goal", "landmark", "midPoint", "pathID", "motion_signalID"]

multi_roles = ["midPoint", "pathID", "motion_signalID"]

link2roles = {
    "QSLINK": ["trigger", "trajector", "landmark"],
    "OLINK": ["trigger", "trajector", "landmark"],
    "MOVELINK": ["trigger", "mover", "source", "goal", "ground", "midPoint", "pathID", "motion_signalID"]
}


class Metrics(Enum):
    OFFICIAL = 1
    STRICT = 2
    DEFAULT = 3
    mSpRL = 4  # 对于mSpRL task, LINK必须有trigger，可以没有trajector或landmark


class Document:
    def __init__(self, path, metric: Metrics, eval_optional_roles=True, eval_null_mover=False,
                 eval_link_attr=False):
        self.path = path
        self.metric = metric
        self.eval_optional_roles = eval_optional_roles
        self.eval_null_mover = eval_null_mover
        self.eval_link_attr = eval_link_attr
        self.element_dict = {}

    def parse_no_trigger(self, element: Element):
        if self.metric == Metrics.OFFICIAL:
            return []

        trigger = element.attrib.get("trigger", "")
        trajector = element.attrib.get("trajector", "")
        landmark = element.attrib.get("landmark", "")
        rel_type = element.attrib.get("relType", "")

        if rel_type == "NEXT TO":
            rel_type = "NEXT_TO"

        link_tuple = (element.tag, "", trajector, landmark)

        if self.eval_link_attr:
            link_tuple += (rel_type,)

        if trajector in self.element_dict and landmark in self.element_dict \
                and trigger not in self.element_dict:
            return [link_tuple]
        return []

    def parse_qs_o_link(self, element: Element):
        trigger = element.attrib.get("trigger", "")
        trajector = element.attrib.get("trajector", "")
        landmark = element.attrib.get("landmark", "")
        rel_type = element.attrib.get("relType", "")

        if rel_type == "NEXT TO":
            rel_type = "NEXT_TO"

        has_trigger = trigger in self.element_dict
        has_two_role = trajector in self.element_dict and landmark in self.element_dict

        if self.metric == Metrics.OFFICIAL and (not has_trigger or not has_two_role):
            return []

        if self.metric == Metrics.STRICT and (not has_two_role):
            return []

        link_tuple = (element.tag,)
        for role in [trigger, trajector, landmark]:
            if role not in self.element_dict:
                role = ""
            link_tuple += (role,)

        if self.eval_link_attr:
            link_tuple += (rel_type,)

        return [link_tuple]

    def parse_move_link(self, element: Element):
        trigger = element.attrib.get("trigger", "")
        mover = element.attrib.get("mover", "")

        if trigger not in self.element_dict:
            trigger = ""
        if mover not in self.element_dict:
            mover = ""

        tuple_ = (element.tag, trigger, mover)

        if not self.eval_null_mover and mover not in self.element_dict:
            return []

        if self.metric == Metrics.OFFICIAL:
            return [] if trigger not in self.element_dict else [tuple_]

        if self.metric == Metrics.STRICT:
            if trigger not in self.element_dict:
                return []

        if self.eval_optional_roles:
            if element.attrib.get("adjunctID", "") != "":
                element.attrib["motion_signalID"] = element.attrib.get("adjunctID")
            for role in move_optional_roles:
                if role in multi_roles:
                    role_ids = element.attrib.get(role, "").replace(" ", "").split(",")
                    role_ids = filter(lambda x: x in self.element_dict, role_ids)
                    role_ids = ",".join(sorted(role_ids))
                    tuple_ += (role_ids,)
                else:
                    role_id = element.attrib.get(role, "")
                    if role_id.__contains__(","):
                        print(role, "ERROR")
                        exit(1)
                    if role_id not in self.element_dict:
                        role_id = ""
                    tuple_ += (role_id,)
        return [tuple_]

    def load_links_from_file(self):
        tree = ElementTree.parse(self.path)
        tags = tree.getroot().find("TAGS")
        link2list = defaultdict(set)

        for element in tags:
            if 'start' in element.attrib and int(element.attrib['start']) != -1 and element.attrib['id'] != "":
                self.element_dict[element.attrib['id']] = int(element.attrib['start'])

        # if "" in element_dict:
        #     exit(1)

        for element in tags:
            if element.tag == "QSLINK" or element.tag == "OLINK":
                # if element.attrib['trajector'] == "pl0":
                #     print()
                link2list[element.tag].update(self.parse_qs_o_link(element))
                link2list["NoTrigger"].update(self.parse_no_trigger(element))
            elif element.tag == "MOVELINK":
                link2list[element.tag].update(self.parse_move_link(element))
        return link2list


def save(input_dir, output_path, eval_optional_roles):
    metric = Metrics.STRICT if eval_optional_roles else Metrics.OFFICIAL

    with open(output_path, "w", encoding="utf-8") as f:
        for root, _, files in os.walk(input_dir):
            for file in files:
                f.write(file + "\n")
                path = os.path.join(root, file)
                document = Document(path=path, metric=metric, eval_optional_roles=eval_optional_roles,
                                    eval_null_mover=False)
                link_dict = document.load_links_from_file()
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
            document = Document(path=path, metric=Metrics.STRICT, eval_optional_roles=True,
                                eval_null_mover=False)
            link_dict = document.load_links_from_file()
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
