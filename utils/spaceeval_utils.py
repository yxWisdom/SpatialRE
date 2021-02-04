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

qs_roles = ["trigger", "trajector", "landmark"]
move_core_roles = ["trigger", "mover"]
move_optional_roles = ["source", "goal", "landmark", "midPoint", "pathID", "motion_signalID"]

multi_roles = ["midPoint", "pathID", "motion_signalID"]


def parse_no_trigger(element: Element, element_dict, eval_null_roles=True):
    trigger = element.attrib.get("trigger", "")
    if element_dict.get(trigger, -1) != -1 or not eval_null_roles:
        return []
    link = ("NoTrigger",)
    for role in qs_roles:
        role_id = element.attrib.get(role, "")
        if element_dict.get(role_id, -1) == -1:
            role_id = ""
        link += (role_id,)
    return [link]


def parse_qs_o_link(element: Element, element_dict, eval_null_roles=True):
    link = (element.tag,)
    for role in qs_roles:
        role_id = element.attrib.get(role, "")
        if element_dict.get(role_id, -1) == -1:
            if not eval_null_roles:
                return []
            role_id = ""
        link += (role_id,)
    return [link]


def parse_move_link(element: Element, element_dict, eval_optional_roles):
    link = (element.tag,)

    # if element.attrib.get("trigger", "") == "":
    #     return []

    for role in move_core_roles:
        role_id = element.attrib.get(role, "")
        if element_dict.get(role_id, -1) == -1:
            role_id = ""
        link += (role_id,)

    if eval_optional_roles:
        for role in move_optional_roles:
            if role in multi_roles:
                role_ids = element.attrib.get(role, "").replace(" ", "").split(",")
                role_ids = filter(lambda x: element_dict.get(x, -1) != -1, role_ids)
                role_ids = ",".join(sorted(role_ids))
                link += (role_ids,)
            else:
                role_id = element.attrib.get(role, "")
                if role_id.__contains__(","):
                    print(role, "ERROR")
                    exit(1)
                if element_dict.get(role_id, -1) == -1:
                    role_id = ""
                link += (role_id,)

    return [link]


# def load_gold_triple(path, eval_optional_roles, eval_null_roles=True):
#     tree = ElementTree.parse(path)
#     root = tree.getroot()
#     tags = root.find("TAGS")
#
#     link_dict = defaultdict(set)
#
#     element_dict = {}
#
#     for element in tags:
#         if 'start' in element.attrib:
#             element_dict[element.attrib['id']] = element.attrib['start']
#     for element in tags:
#         if element.tag == "QSLINK" or element.tag == "OLINK":
#             link_dict[element.tag].update(parse_qs_o_link(element, element_dict, eval_null_roles))
#             link_dict["NoTrigger"].update(parse_no_trigger(element, element_dict, eval_null_roles))
#         elif element.tag == "MOVELINK":
#             link_dict[element.tag].update(parse_move_link(element, element_dict, eval_optional_roles))
#     return link_dict

def load_gold_triple(path, eval_optional_roles, eval_null_roles=True):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    tags = root.find("TAGS")

    link_dict = defaultdict(list)

    element_dict = {}

    for element in tags:
        if 'start' in element.attrib:
            element_dict[element.attrib['id']] = int(element.attrib['start'])
    for element in tags:
        if element.tag == "QSLINK" or element.tag == "OLINK":
            link_dict[element.tag].extend(parse_qs_o_link(element, element_dict, eval_null_roles))
            link_dict["NoTrigger"].extend(parse_no_trigger(element, element_dict, eval_null_roles))
        elif element.tag == "MOVELINK":
            link_dict[element.tag].extend(parse_move_link(element, element_dict, eval_optional_roles))
    return link_dict


if __name__ == '__main__':
    import os

    all_link_dict = defaultdict(int)
    data_dir = "D:\\项目\\军事问答\\project\\pytorch_nlp\\dataset\\MHS\\sample_5\\xml"
    # data_dir = "D:\\项目\空间关系识别\\repo\\spatialie\\data\\SpaceEval2015\\raw_data\\training++"
    # data_dir = "D:\\项目\\空间关系识别\\repo\\spatialie\\data\\SpaceEval2015\\predict_result\\MHS\\configuration3\\full\\XML"
    # for root, _, files in os.walk(data_dir):
    #     for file in files:
    #         print()
    #         print(file)
    #         path = os.path.join(root, file)
    #         link_dict = load_gold_triple(path, True, True)
    #         for link_type, links in link_dict.items():
    #             print(link_type, len(links))
    #             all_link_dict[link_type] += len(links)
    # print(all_link_dict)

    # data_dir = "F:\\垃圾场\\SpatialRelEx\\output"

    for root, _, files in os.walk(data_dir):
        for file in files:
            print()
            print(file)
            path = os.path.join(root, file)
            d = load_gold_triple(path, True, True)
            for d, links in d.items():
                # if d != "MOVELINK":
                #     continue
                print(d)
                links.sort()
                for link in links:
                    print(link)
                    # tuples = eval(link[-3])
                    # if len(eval(link[-3])) > 1 or len(eval(link[-2])) > 1: #or len(eval(link[-1]))>1:
                    #     print(link)

    # path = "D:\\项目\\军事问答\\project\\pytorch_nlp\\dataset\\MHS\\sample_5\\xml\\48_N_10_E.xml"
    # d = load_gold_triple(path, True, True)
    # for d, links in d.items():
    #     print(d)
    #     for link in links:
    #         print(link)
