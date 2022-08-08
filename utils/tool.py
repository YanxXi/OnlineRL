import numpy as np
import pandas as pd
from collections import Counter

def mean_squared_error(output, target):
    '''
    calculate the mean squared error (MSE) given target and predicted output
    '''
 
    target = np.array(target) if type(target) != np.ndarray else target
    output = np.array(output) if type(output) != np.ndarray else output

    R = target - output

    mse = np.power(R, 2).mean()

    return mse

def r2_score(output, target):
    target = np.array(target) if type(target) != np.ndarray else target
    output = np.array(output) if type(output) != np.ndarray else output
    
    r2 = 1 - mean_squared_error(output, target) / np.var(target)
    
    return r2


def cal_accuracy(output, target):
    '''
    calculate the classification accuracy
    '''
    
    assert len(output) == len(target)
    output = np.array(output) if not isinstance(output, np.ndarray) else output
    target = np.array(target) if not isinstance(target, np.ndarray) else target
    accuracy = np.equal(output, target).sum() / len(output)

    return accuracy


def train_test_split(data_set:np.ndarray, shuffle_data=True, train_size=0.75):
    '''
    split the data set to train set and test set
    '''
    
    if shuffle_data:
        np.random.shuffle(data_set)
    train_data_set, test_data_set =  np.split(data_set, [int(len(data_set) * train_size)])

    return train_data_set, test_data_set



def predict(vector, tree):
    '''
    predict the value of one sample based on the given decision tree
    '''
    
    # not leaf node
    if isinstance(tree, dict):
        feature = list(tree.keys())[0]
        split_dict = tree[feature]
        split_value = float(list(split_dict.keys())[0][2:])

        if vector[feature] <= split_value:
            sub_tree = split_dict['<=' + str(split_value)]
            output = predict(vector, sub_tree)
        else:
            sub_tree = split_dict['>' + str(split_value)]
            output = predict(vector, sub_tree)
    # leaf node
    else:
        return tree

    return output


def tree_predict_(data_set:pd.DataFrame, tree:dict):
    '''
    predict the value of the data set based on the given decision tree
    '''
   
    tree_output = [predict(vec, tree) for (idx, vec) in data_set.iterrows()]
  

    return np.array(tree_output)

def cal_num_leaf(tree):
    '''
    calculate the number of leafs of the given tree
    '''

    num_leaf = 0
    split_dict = tree[list(tree.keys())[0]]

    for key in split_dict.keys():
        sub_tree = split_dict[key]
        if type(sub_tree).__name__ == 'dict':
            num_leaf += cal_num_leaf(sub_tree)
        else:
            num_leaf += 1

    return num_leaf


def classify(vector, tree):
    '''
    classfy one sample based on the given decision tree
    '''
    
    # not leaf node
    if isinstance(tree, dict):
        feature = list(tree.keys())[0]
        split_dict = tree[feature]
        split_value = float(list(split_dict.keys())[0][2:])

        if vector[feature] <= split_value:
            sub_tree = split_dict['<=' + str(split_value)]
            output = classify(vector, sub_tree)
        else:
            sub_tree = split_dict['>' + str(split_value)]
            output = classify(vector, sub_tree)
    # leaf node
    else:
        return tree

    return output


def tree_classify_(data_set:pd.DataFrame, tree):
    '''
    classify the data set based on the given decision tree

    Args:
        tree: DecisionTreeClassifier

    Returns:
        tree_output: result of classification
        
    '''

    
    tree_output = [classify(vec, tree) for (idx, vec) in data_set.iterrows()]


    return np.array(tree_output)


def split_dataset(data_set, split_feature, split_value) -> list:
    '''
    split dataset according to given axis (feature index) and value (split value)
    '''

    left_dataset =  data_set[data_set[split_feature] <= split_value]
    right_dataset = data_set[data_set[split_feature] > split_value]

    return [left_dataset, right_dataset]


def major_vote(targets):

    '''
    decide the class in class list according to mojority vote rule
    '''


    counter = Counter(targets)
    
    return counter.most_common(1)[0][0]


def get_leaf(tree):
    '''
    get all leaf value of the tree
    '''

    leaf_accumulator = []
    if isinstance(tree, dict):
        feature = list(tree.keys())[0]
        split_dict = tree[feature]
        split_value = float(list(split_dict.keys())[0][2:])
        left_tree = split_dict['<=' + str(split_value)]
        left_leaf_accumulator = get_leaf(left_tree)
        right_tree = split_dict['>' + str(split_value)]
        right_leaf_accumulator = get_leaf(right_tree)
        leaf_accumulator.extend(left_leaf_accumulator)
        leaf_accumulator.extend(right_leaf_accumulator)
    else:
        leaf_accumulator.append(tree)

    return leaf_accumulator

