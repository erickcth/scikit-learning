import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

# for i in range(len(iris.data)):
# 	print("Iris %d: label: %s, features %s " % (i, iris.target[i], iris.data[i]))

test_idx = [0, 50, 100]

# training
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# visualise
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")


# testing
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

print(iris.feature_names, iris.target_names)
print(test_data)
print(test_target)
print(clf.predict(test_data))
