import numpy as np
import os
import graphviz
import math


def entropy(y):
    """
    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    h = 0
    values , count = np.unique(y,return_counts="True")
    for c in count:
        h+=c/len(y)*(math.log((c/len(y)),2))
    return -h;


def mutual_information(x, y):
    """
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    mut_info = {}
    entropy_y = entropy(y)
    len_y = len(y)
    values , count = np.unique(x,return_counts="True")
    for v in values :
        new_y1 = np.array(y)[np.where(x==v)[0]]
        new_y2 = np.array(y)[np.where(x!=v)[0]]
        mut_info[v] = round(entropy_y - (((len(new_y1)/len_y)*entropy(new_y1))+((len(new_y2)/len_y)*entropy(new_y2))),7)
    return mut_info


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    if(all(items == y[0] for items in y)):
        return y[0]
    if(depth>0 and len(attribute_value_pairs) == 0):
       return max(set(y), key=y.count)
    if depth <= max_depth:
        attribute_vals = []
        mutualInformation = []
        
        for i in range(0,len(x[0])) :
                attribute_vals.insert(i, [row[i] for row in x])
        if depth == 0:
            attribute_value_pairs =[]            
            for i in range(0,len(attribute_vals)) :
                values = np.unique(attribute_vals[i])
                for val in values :
                    pairs = [i,val]
                    attribute_value_pairs.append(pairs)
        
        for i in range(0,len(attribute_vals)):
            mutualInformation.insert(i, mutual_information(attribute_vals[i], y))
            
        max_attr_val_pair = 0 
        max_MI = 0;
        
        for i in range(0,len(mutualInformation)) :
            max_attr_key = max(mutualInformation[i],key=mutualInformation[i].get)
            max_attr_value = max(mutualInformation[i].values())
            if (max_attr_value > max_MI):
                max_attr_val_pair = [i,max_attr_key]
                max_MI = max_attr_value
                
        newx1 = []
        newy1 = []
        newx2 = []
        newy2 = []
        
        for i in range(0,len(x)):
            x_row=x[i]
            if(x_row[max_attr_val_pair[0]]==max_attr_val_pair[1]):
                newx1.append(x[i])
                newy1.append(y[i])
            else :
                newx2.append(x[i])
                newy2.append(y[i])
          
        new_attribute_value_pairs = [i for i in attribute_value_pairs if i != max_attr_val_pair]
        
        final_dict = {}
        depth+=1
        tupple1 = (max_attr_val_pair[0],max_attr_val_pair[1],False)
        tupple2 = (max_attr_val_pair[0],max_attr_val_pair[1],True)
        final_dict[tupple1] = id3(newx2,newy2,new_attribute_value_pairs,depth,max_depth)
        final_dict[tupple2] = id3(newx1,newy1,new_attribute_value_pairs,depth,max_depth)
        
        return final_dict
    else :
        return max(set(y), key=y.count)
    
        


def predict_example(x, tree):
    """
    Returns the predicted label of x according to tree
    """
    if (tree==0 or  tree==1):
        return tree
    node = list(tree.keys())[0]
    if(x[node[0]]==node[1]):
        return predict_example(x,tree.get((node[0],node[1],True)))
    else:
        return predict_example(x,tree.get((node[0],node[1],False)))


def compute_error(y_true, y_pred):
    """
    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    return sum(y_true != y_pred)/len(y_true)



def pretty_print(tree, depth=0):
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
