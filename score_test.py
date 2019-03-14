import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
import time

from iforest import IsolationTreeEnsemble, find_TPR_threshold

def score(X, y, n_trees, desired_TPR, datafile,sample_size,
          reqd_fit_time,
          reqd_score_time,
          reqd_FPR,
          reqd_n_nodes):
    it = IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)

    fit_start = time.time()
    it.fit(X, improved=improved)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    print(f"INFO {datafile} fit time {fit_time:3.2f}s")

    n_nodes = sum([t.n_nodes for t in it.trees])
    print(f"INFO {datafile} {n_nodes} total nodes in {n_trees} trees")

    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    print(f"INFO {datafile} score time {score_time:3.2f}s")

    threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)

    y_pred = it.predict_from_anomaly_scores(scores, threshold=threshold)
    confusion = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    errors = 0
    if fit_time > reqd_fit_time * 2:
        print(f"FAIL {datafile} fit time {fit_time:.1f} > {reqd_fit_time}")
        errors += 1

    if score_time > reqd_score_time * 2:
        print(f"FAIL {datafile} score time {score_time:.1f} > {reqd_score_time}")
        errors += 1

    if TPR < desired_TPR*.9: # TPR must be within 10% (or above)
        print(f"FAIL {datafile} TPR {TPR:.2f} < {desired_TPR} +- 10%")
        errors += 1

    if FPR > reqd_FPR*1.3: # TPR must be within 30%
        print(f"FAIL {datafile} FPR {FPR:.4f} > {reqd_FPR} +- 30%")
        errors += 1

    if n_nodes > reqd_n_nodes*1.15:
        print(f"FAIL {datafile} n_nodes {n_nodes} > {reqd_n_nodes} +- 15%")
        errors += 1

    if errors==0:
        print(f"SUCCESS {datafile} {n_trees} trees at desired TPR {desired_TPR*100.0:.1f}% getting FPR {FPR:.4f}%")
    else:
        print(f"ERRORS {datafile} {errors} errors {n_trees} trees at desired TPR  {desired_TPR*100.0:.1f}% getting FPR {FPR:.4f}%")


def add_noise(df):
    n_noise = 5
    for i in range(n_noise):
        df[f'noise_{i}'] = np.random.normal(0,100,len(df))


def score_test
    # use any dataset
    df = pd.read_csv("data.csv")
    N = len(df)
    df = df.sample(N)  # grab random subset
    if noise: add_noise(df)
    X, y = df.drop('label', axis=1), df['label']

    score(X, y, n_trees=1000, desired_TPR=.75,sample_size=5,
          datafile='cancer.csv',
          reqd_fit_time=0.2,
          reqd_score_time=.75,
          reqd_FPR=.33,
          reqd_n_nodes=8500)


if __name__ == '__main__':
    noise = False
    improved = False
    if '-noise' in sys.argv:
        noise = True
    if '-improved' in sys.argv:
        improved = True

    print(f"Running noise={noise} improved={improved}")
    score_test()
    print()
