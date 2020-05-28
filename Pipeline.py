from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# using decission tree classifier
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)

# using k-neighbors classifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, predictions), "%")
