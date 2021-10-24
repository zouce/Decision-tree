import pandas
import pandas as pd
import numpy as np
import math
import pprint
import copy
from learning_lib import train_test_split

pp = pprint.PrettyPrinter(indent=4)


def load_data():
    # Only include the first 8 descriptive features and the target label
    data = pd.read_csv("heart.csv", usecols=[
                       "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"])
    return data


def describe_partitions(ps):
    for target, p in sorted(ps.items(), key=lambda k: k[0]):
        print(f"{target}\t{p.shape[0]}")
    print("")


# 算熵
def entropy(data):
    counts = data["target"].value_counts()
    """
        Similar to doing the following manually:
            counts = {}
            for val in data["target"]:
                counts[val] = counts.get(val, 0) + 1
    """
    total = data["target"].shape[0]
    sum = 0.
    for count in counts:
        p = count/total
        sum += p * math.log(p)
    return - sum


def partitions(data, feature, thresholds):
    def find_threshold(feature, val):
        # Guaranteed to find a threshold somewhere between min and max
        for t in reversed(thresholds[feature]):
            if val >= t:
                return t
        raise Exception("Unexpected return without threshold")

    features = data.columns
    ps = {}
    for j, val in enumerate(data[feature]):
        # Treat categorical and continuous feature values differently
        if feature in thresholds:
            val = find_threshold(feature, val)
        p = ps.get(val, pd.DataFrame(columns=features))
        ps[val] = p.append(data.loc[j, features])
    return ps


def create_thresholds(data, names, nstds=3):
    if len(names) == 0:
        return {}
    # Assume the data is normally-distributed
    thresholds = {}
    for feature in names:
        col = data[feature]
        mint, maxt = np.min(col), np.max(col)
        mean, stddev = np.mean(col), np.std(col)
        ts = [mint]
        for n in range(-nstds - 1, nstds):
            if feature == "oldpeak":
                t = round(n * stddev + mean, 5)
            else:
                t = round(n * stddev + mean)
            if mint < t <= maxt:
                ts.append(t)
        thresholds[feature] = ts
    return thresholds


def gain(data, H, feature, thresholds):
    ps = partitions(data, feature, thresholds)
    # describe_partitions(ps)
    sum = 0.
    for p in ps.values():
        if feature in p.columns:
            sum += (p.shape[0] / data.shape[0]) * entropy(p)
    return H - sum


def data_split(father, thresholds, thresholds1, data):
    d = []
    if father in thresholds:
        # print(data)
        last = 10000
        for i in range(len(thresholds[father])):
            d.append({})
        for i in range(len(thresholds[father])):
            ii = len(thresholds[father])-1-i
            for item in data.columns:
                if item == father:
                    continue
                d[i][item] = []
            # print(thresholds[father][ii], last)
            for j in range(len(data["target"])):
                if thresholds[father][ii] <= data[father][j] < last:
                    for k in d[i].keys():
                        if k == "index":
                            d[i][k].append(len(d[i][k])+1)
                        else:
                            d[i][k].append(data[k][j])
            last = thresholds[father][ii]
    else:
        for i in range(len(thresholds1[father])):
            d.append({})
            for item in data.columns:
                if item == father:
                    continue
                d[i][item] = []
            for j in range(len(data["target"])):
                if data[father][j] == thresholds1[father][i]:
                    for k in d[i].keys():
                        if k == "index":
                            d[i][k].append(len(d[i][k])+1)
                        else:
                            d[i][k].append(data[k][j])
    return d


def find_next(father, thresholds, thresholds1, data, size):
    if size == 1 or len(data["target"]) == 0:
        pass
    counts = (list(data["target"])).count(0)
    total = len(data["target"])
    # print(counts, "=====", total)
    if counts == total:
        return 0
    if counts == 0:
        return 1
    d = data_split(father, thresholds, thresholds1, data)
    thresholds_list = []
    for i in ["age", "chol", "trestbps", "thalach", "oldpeak"]:
        if i in d[0].keys():
            thresholds_list.append(i)
    res = {}
    t = 0
    for i in d:
        if len(i["target"]) == 0:
            continue
        # print(i)
        # print(father, size)
        new = pandas.DataFrame(i)
        H = entropy(new)
        thresholds_new = create_thresholds(
            new, thresholds_list, nstds=3)
        IG = np.zeros(size-1)
        for j, feature in enumerate(new.columns[1:size]):
            IG[j] = gain(new, H, feature, thresholds_new)

        son = new.columns[np.argmax(IG)+1]
        if father in thresholds:
            res[str(thresholds[father][t])] = find_next(son, thresholds_new, thresholds1, new, size-1)
        else:
            res[str(thresholds1[father][t])] = find_next(son, thresholds_new, thresholds1, new, size-1)
        t += 1
    return {father: res}


def judge(data, index, keys, k):
    for i in range(len(keys)):
        j = len(keys)-1-i
        if data[k][index] >= float(keys[j]):
            return keys[j]
    return keys[0]


def test(data, index, tree, thresholds):
    key = list(tree.keys())[0]
    if key in thresholds:
        t = tree[key][judge(data, index, list(tree[key].keys()), key)]
    else:
        if str(data[key][index]) in tree[key].keys():
            t = tree[key][str(data[key][index])]
        else:
            # t = tree[key][list(tree[key].keys())[0]]
            return 1
    if type(t) == int:
        return t
    else:
        return test(data, index, t, thresholds)


def main():
    data = load_data()
    # Split into training and test data sets
    train_data, test_data = train_test_split(data, test_size=0.25)
    # Compute the total entropy for the full data set with respect to the target label
    H = entropy(train_data)
    print(f"Total Entropy: {H}")
    # Generate threshold values for the continuous value descriptive features
    thresholds = create_thresholds(
        train_data, ["age", "chol", "trestbps", "thalach", "oldpeak"], nstds=3)
    print(thresholds)
    thresholds1 = {'sex': [0, 1], 'cp': [0, 1, 2, 3], 'fbs': [0, 1], 'restecg': [0, 1, 2], 'exang': [0, 1], 'slope': [0, 1, 2], 'ca': [0, 1, 2, 3, 4], 'thal': [0, 1, 2, 3]}
    # Compute the level=0 information gain when partitioned on each descriptive feature
    # print(thresholds)
    IG = np.zeros(13)
    for i, feature in enumerate(data.columns[:13]):
        IG[i] = gain(train_data, H, feature, thresholds)
    # Print the best one (at the level=0)
    print(IG)
    father = data.columns[np.argmax(IG)]
    print(f"Best IG feature: {father}")
    tree = find_next(father, thresholds, thresholds1, train_data, 13)
    # print(tree)
    ans = 0
    total = len(test_data["target"])
    for i in range(total):
        res = test(test_data, i, tree, thresholds)
        if res == data["target"][i]:
            ans += 1
    print(ans/total)


if __name__ == "__main__":
    main()
