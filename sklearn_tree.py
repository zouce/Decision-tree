from sklearn import metrics
from sklearn import tree
import pandas as pd
import graphviz
from learning_lib import train_test_split

# Need to update path to graphviz
import os
os.environ["PATH"] += os.pathsep + u"C:\\Users\\brian\\Anaconda3\\envs\\intro-ml\\Library\\bin\\graphviz"


data = pd.read_csv("heart.csv", usecols=[
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "target"])
train_data, test_data = train_test_split(data, test_size=0.15)
X_train, y_train = train_data.loc[:, "age":"thalach"], train_data["target"]
X_test, y_test = test_data.loc[:, "age":"thalach"], test_data["target"]

"""
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
"""

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(model.score(X_test, y_test))

graph_data = tree.export_graphviz(
    model, out_file=None, feature_names=data.columns[:8], filled=True)
graph = graphviz.Source(graph_data)
graph.render("data", view=True)
