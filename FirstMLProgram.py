from sklearn import tree

# training data
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
lables = [0, 0, 1, 1]

# creating the classifier
clf = tree.DecisionTreeClassifier()

# initializing the training algorithm
clf = clf.fit(features, lables)

# input a new value to get a predicction
print(clf.predict([[150, 0]]))

# this will give the prediction as [1]
