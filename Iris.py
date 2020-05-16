from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()

# initializing seperating index
test_index = [0, 50, 100]

# seperating data to test
train_data = np.delete(iris.data, test_index, axis=0)

# seperating targets to test
train_target = np.delete(iris.target, test_index)

# training portion of data
test_data = iris.data[test_index]

# training portion of targets
test_target = iris.target[test_index]

# initializing the classifier and the training algorithm
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# testing predictions
print("Actual result : ", test_target)
print("Predicted result : ", clf.predict(test_data))
