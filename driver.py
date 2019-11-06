from DecisionTree import *
import pandas as pd
from sklearn import model_selection

divider = "***************"
header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print("\n" + divider + " Tree before splitting dataset into training and test data " + divider)
print_tree(t)

print("\n" + divider + " Leaf nodes " + divider)
# Get leaf nodes and print
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth = " + str(leaf.depth))

print("\n" + divider + " Non-leaf nodes " + divider)
# Get non-leaf nodes and print
innerNodes = getInnerNodes(t)
for node in innerNodes:
    print("id = " + str(node.id) + " depth =" + str(node.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.3)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("\n" + divider + " Tree on train data before pruning " + divider)
print_tree(t)
acc = computeAccuracy(test, t)
print(divider + " Accuracy on test = " + str(acc) + " " + divider)

# ## TODO: You have to decide on a pruning strategy
idList = getInnerNodesId(t)
if 0 in idList:
    idList.remove(0)
max_acc = acc
print("\n List of ids of nodes 1 level above the leaves = " + str(idList))
for id in idList:
    t_pruned = prune_tree(t, [id])
    acc_pruned = computeAccuracy(test, t_pruned)
    print("\n" + divider + " Tree after pruning Node "+str(id) + " " + divider)
    print_tree(t_pruned)
    print(divider + " Accuracy on test after pruning node " + str(id) + " = " + str(acc_pruned) + " " + divider)
    if acc_pruned > max_acc:
        max_acc = acc_pruned
    if acc_pruned < max_acc:
        print("Pruning stopped as accuracy dropped")
        break




