
import pandas as pd
import numpy as np
import os
import math

# split the given dataset on the feature given

def train_split(dataset, feature, threshold):
    data_vec = list()
    if "{n}" in feature:   # nominal
        feature_vals = dataset[feature].unique()       
        feature_vals.sort()      
        if len(feature_vals) < 2:
            return None

        for val in feature_vals:
            data_vec.append(dataset.loc[dataset[feature] == val])

    else:               
        less = dataset.loc[dataset[feature] < threshold]
        if len(less):
            data_vec.append(less)
        else:
            return None

        greater_eq = dataset.loc[dataset[feature] >= threshold]
        if len(greater_eq):
            data_vec.append(greater_eq)
        else:
            return None
    return data_vec


# calculate the information gain of a given split dataset
def inform_gain(data_vec):
    total_rows = 0.0
    for subframe in data_vec:
        total_rows += len(subframe.index)      
    p_target_vals = list()
    total_index = np.concatenate([(sub.index) for sub in data_vec])
    unique_target_vals, target_val_freq = np.unique(total_index,
            return_counts=True)
    for n in range(0,len(unique_target_vals)):
        p_target_vals.append(target_val_freq[n] / sum(target_val_freq))

    if (p_target_vals[0] == 1):
        return 0       

    entropy = 0
    for p in p_target_vals:
        entropy += -(p)*math.log(p,2)


    conditional_entropy = 0
    for n, subframe in enumerate(data_vec):
        subframe_size = len(subframe.index)
        subframe_prob = subframe_size / total_rows

        conditional_prob = 0
        for val in [x for x in unique_target_vals if x in subframe.index]:
            val_freq = len([x for x in subframe.index == val if x == True])
            p = val_freq / subframe_size
            conditional_prob += -p*math.log(p,2)

        conditional_entropy += subframe_prob*conditional_prob

    inform_gain = entropy - conditional_entropy
    if (inform_gain >= 0):
        return inform_gain
    else:
        raise Exception("Something went wrong... Information Gain is negative")


# process a terminal node once one of the terminal conditions is hit
def terminate_node(dataset):
    likely_class = None
    freq = 0
    for class_val in dataset.index.unique():
        if len([x for x in dataset.index == class_val if x == True]) > freq:
            likely_class = class_val

    return likely_class


# split with lowest information gain
def calc_best_split(dataset):
    feature, threshold, inf_gain, data_vec = None, np.array([0]), 0, None
    for col in dataset.columns:
        if "{n}" in col:   
            data_vec_temp = train_split(dataset, col, t)
            if data_vec_temp == None:
                continue

            inf_gain_temp = inform_gain(data_vec_temp)

            if inf_gain_temp > inf_gain:
                inf_gain = inf_gain_temp
                data_vec = data_vec_temp
                feature = col
                feature_vals = dataset[col].unique()
                feature_vals.sort()
                threshold = feature_vals
            else:
                continue

        else:              
            for t in dataset[col].unique():
                data_vec_temp = train_split(dataset, col, t)
                if data_vec_temp == None:
                    continue

                inf_gain_temp = inform_gain(data_vec_temp)
                if inf_gain_temp > inf_gain:
                    inf_gain = inf_gain_temp
                    data_vec = data_vec_temp
                    feature = col
                    threshold[0] = t
                else:
                    continue

    if feature == None:
        return terminate_node(dataset)
    else:
        return {'feature':feature, 'value':threshold, 'data':data_vec}



# recursive splitting function used to split on them splits
def recursive_split(node, max_depth, min_size, current_depth):

    print("Feature: {}\tValues: {}\tData Len: {}".format(node['feature'],
            node['value'], len(node['data'])))

    children_data = list()
    for data in node['data']:
        children_data.append(data)

    del(node['data'])

    num_children = len(children_data)
    node['children'] = {}

    print("Depth: {}\tChildren: {}".format(current_depth, num_children))
    if current_depth >= max_depth:
        for n in range(num_children):
            node['children'][node['value'][n]] = terminate_node(children_data[n])
        return
    if len(node['value']) > 1:         
        for child in range(num_children):
            if len(children_data[child]) > min_size:
                node['children'][node['value'][child]] = calc_best_split(
                        children_data[child])
                if type(node['children'][node['value'][child]]) == dict:
                    recursive_split(node['children'][node['value'][child]],
                            max_depth, min_size, current_depth+1)
            else:
                node['children'][node['value'][child]] = terminate_node(
                        children_data[child])
    else:
        if len(children_data[0]) > min_size:
            node['children']['less'] = calc_best_split(children_data[0])
            if type(node['children']['less']) == dict:  
                recursive_split(node['children']['less'],
                        max_depth, min_size, current_depth+1)
            else:          
                node['children']['less'] = terminate_node(children_data[0])
        else:
            node['children']['less'] = terminate_node(children_data[0])

        if len(children_data[1]) > min_size:
            node['children']['greater'] = calc_best_split(children_data[1])
            if type(node['children']['greater']) == dict:
                recursive_split(node['children']['greater'],
                        max_depth, min_size, current_depth+1)
            else:          
                node['children']['greater'] = terminate_node(children_data[1])
        else:
            node['children']['greater'] = terminate_node(children_data[1])

    return
    
#build the tree
def build_decision_tree(dataset, max_depth, min_size):

    root = calc_best_split(dataset)
    recursive_split(root, max_depth, min_size, 1)
    return root

def print_tree(node, depth):
    if isinstance(node, dict):
        if "{n}" in node['feature']:
            for n,child in enumerate(node['children']):
                print('%s[%s = %.3f]' % ((depth*' ',
                    (node['feature']),node['value'][n])))
                print_tree(node['children'][child], depth+1)
        else:                          
            print('%s[%s < %.3f]' % ((depth*' ',
                (node['feature']),node['value'][0])))
            for child in node['children']:
                print_tree(node['children'][child], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))






