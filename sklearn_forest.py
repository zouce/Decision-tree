from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from learning_lib import train_test_split

data = pd.read_csv("heart.csv", usecols=[
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "target"])
train_data, test_data = train_test_split(data, test_size=0.15)
X_train, y_train = train_data.loc[:, "age":"thalach"], train_data["target"]
X_test, y_test = test_data.loc[:, "age":"thalach"], test_data["target"]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = RandomForestClassifier(n_estimators = 1000, random_state = 42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(model.score(X_test, y_test))
