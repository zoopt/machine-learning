#!/urs/bin/python
#-*- coding:utf-8 -*-

import sys
import math
import operator



def split_data_set(data_set, attr, value):
    """
    find all data_set[:][attr]=value samples
    """
    # check parameters
    if (not data_set) or not (attr>=0 and attr<len(data_set[0])):
        return []
    ret = []
    for item in data_set:
        if item[attr] == value:
            record = item[:attr] # remove attr
            record += item[attr+1:]
            ret.append(record)
    return ret

def calc_entropy(data_set):
    """
    cal entropy
    """
    if (not data_set):
        return 0.0
    item_num = len(data_set)
    label_counts = {}
    for item in data_set:
        label = item[-1]
        label_counts[label] = label_counts.setdefault(label, 0)+1
    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/item_num
        entropy += (-1.0)*prob*math.log(prob, 2)
    return entropy

def select_best_attr(data_set):
    """
    find the best attr to split
    """
    # check parameter
    if (not data_set):
        return -1
    # the last col is class label
    attr_num = len(data_set[0])-1
    base_entropy = calc_entropy(data_set)
    best_info_gain = 0.0
    best_attr = -1
    for attr_idx in range(attr_num):
        attr_value = [item[attr_idx] for item in data_set]
        attr_uniq_value = set(attr_value)
        joint_entropy = 0.0
        for attr_value in attr_uniq_value:
            sub_data_set = split_data_set(data_set, attr_idx, attr_value)
            prob = float(len(sub_data_set))/len(data_set)
            joint_entropy += prob*calc_entropy(sub_data_set)
        info_gain = base_entropy- joint_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_attr = attr_idx

    return best_attr

def major_vote_class(class_list):
    """
    calc the most freq class
    """
    if (not data_set):
        return -1
    stat_dict = {}
    for vote in class_list:
        stat_dict[vote] = stat_dict.setdefault(vote, 0)+1
    class_sorted = sorted(stat_dict.iteritems(), \
            key=operator.itemgetter(1), reverse=True)
    return class_sorted[0][0]

def create_id3(data_set, labels):
    """
    train decision tree id3 model
    """
    # check parameter,for invalid parameter return empty tree
    if (not data_set) or (not isinstance(data_set, list)):
        return {}

    class_list = [item[-1] for item in data_set]
    # stop splitting when all sample in same class
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return major_vote_class(class_list)
    best_attr = select_best_attr(data_set)
    best_attr_label = labels[best_attr]
    id3_tree = {best_attr_label:{}}
    del labels[best_attr]
    attr_values = [item[best_attr] for item in data_set]
    attr_uniq_values = set(attr_values)
    for attr_value in attr_uniq_values:
        sub_labels = labels[:]
        id3_tree[best_attr_label][attr_value] = create_id3(\
                split_data_set(data_set,best_attr, attr_value), \
                sub_labels)
    return id3_tree

def classify(id3_tree, attr_labels, test_vec):
    """
    classify your sample
    """
    if (not id3_tree) or (not isinstance(id3_tree,dict)):
        return "ERROR"

    root_attr = id3_tree.keys()[0]
    second_dict = id3_tree[root_attr]
    attr_idx = attr_labels.index(root_attr)
    if attr_idx != -1:
        key = test_vec[attr_idx]
    else:
        return "ERROR"
    value = second_dict[key]
    if isinstance(value, dict):
        class_label = classify(value, attr_labels, test_vec)
    else:
        class_label = value
    return class_label

def push_tree(id3_tree, file_name):
    """
    store id3 tree on disk
    """
    import pickle
    fw = open(file_name, "w")
    pickle.dump(id3_tree, fw)
    fw.close()

def pull_tree(file_name):
    """
    load id3 tree from disk
    """
    import pickle
    fr = open(file_name)
    return pickle.load(fr)

def create_demo():
    data_set = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return data_set, labels

