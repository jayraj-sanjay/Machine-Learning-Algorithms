import numpy as np
import os
import graphviz
import math
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


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
        
        new_attribute_value_pairs= [i for i in attribute_value_pairs if i != max_attr_val_pair]
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



def learning_curve(X_train,Y_train,X_test,Y_test):
    dec_tree = []
    tst_err = []
    train_err = []
    for i in range(1,11):
        dec_tree.append(id3(X_train, Y_train, max_depth=i))
    for i in range(0,10):
        y_pred_test = [predict_example(x, dec_tree[i]) for x in X_test]
        tst_err.append((compute_error(Y_test, y_pred_test) *100).round(2))
        
        y_pred_train = [predict_example(x, dec_tree[i]) for x in X_train]
        train_err.append((compute_error(Y_train, y_pred_train) *100).round(2))
    
    plt.figure()
    plt.plot(list(range(1,11)), list(tst_err), marker='o', linewidth=3, markersize=12)
    plt.plot(list(range(1,11)), list(train_err), marker='s', linewidth=3, markersize=12)
    plt.xlabel('Depth', fontsize=16)
    plt.ylabel('Test error', fontsize=16)
    plt.xticks(list(range(0,12)), fontsize=12)
    plt.legend(['Test Error', 'Train Error'], fontsize=10)
    plt.axis([0, 12, 0, 60])
    
def weak_learner(X_train,Y_train,X_test,Y_test):
    for i in range(1,6,2):
        dec_tree = id3(X_train, Y_train, max_depth=i)
        y_pred = [predict_example(x, dec_tree) for x in X_test]
        print_confusion_matrix(Y_test,y_pred)

def print_confusion_matrix(y_test,y_pred):
    cm = confusion_matrix(ytst1, y_pred, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['0','1'])
    cmd.plot()  
    
def tree_classifier(X_train,Y_train,X_test,Y_test):
    for i in range(1,6,2):
        clf = DecisionTreeClassifier(criterion="entropy",max_depth=i)
        clf = clf.fit(Xtrn1,ytrn1)
        y_pred = clf.predict(Xtst1)
        print_confusion_matrix(ytst1,y_pred)
    
    
if __name__ == '__main__':
        
    # Load the training data 1
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn1 = M[:, 0]
    Xtrn1 = M[:, 1:]

    # Load the test data 1
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst1 = M[:, 0]
    Xtst1 = M[:, 1:]
    
    # Load the training data 2
    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn2 = M[:, 0]
    Xtrn2 = M[:, 1:]
    
    # Load the test data 2
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst2 = M[:, 0]
    Xtst2 = M[:, 1:]
    
    # Load the training data 3
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn3 = M[:, 0]
    Xtrn3 = M[:, 1:]
    
    # Load the test data 3    
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst3 = M[:, 0]
    Xtst3 = M[:, 1:]
    
    
    learning_curve(Xtrn1,ytrn1,Xtst1,ytst1)
    learning_curve(Xtrn2,ytrn2,Xtst2,ytst2)
    learning_curve(Xtrn3,ytrn3,Xtst3,ytst3)
    
    weak_learner(Xtrn1,ytrn1,Xtst1,ytst1)
    
    tree_classifier(Xtrn1,ytrn1,Xtst1,ytst1)
        
    M = np.genfromtxt('./hayes-roth.data', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y = M[:, -1]
    X = M[:, 1:5]
    values = np.unique(y)
    mean = sum(values)/len(values)
    y = np.where(y <= mean, 0, y)
    y = np.where(y > mean, 1, y)
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, random_state=42)
    tree_classifier(X_trn,y_trn,X_tst,y_tst)  
