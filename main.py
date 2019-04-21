import datetime

import h5py
import numpy as np
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

### DATA READING STAGE ###

f = h5py.File("data/BJ16_M32x32_T30_InOut.h5", "r")

# Transform a dataset in a .h5 file into an numpy array.

date = f["date"][:]
# A array of shape 7220x1 where date[i] is the time of i-th time slot.

data = f["data"][:]
# An array of shape 7220x2x32x32 where the first dimension represents the index of time slots. data[i][0] is a (32,
# 32) inflow matrix and data[i][1] is a (32, 32) outflow matrix.

f.close()

### DATA PREPROCESSING STAGE ###

dateReadable = np.zeros((date.size, 3), dtype=int)
# Column 1: The number of days from the INITIAL_DATE to the current one
# Column 2: The number of time slot. Each time slot lasts 30 minutes.
# Column 3: "1" if this date is in weekend. "0" otherwise.

INITIAL_DATE = datetime.date(2015, 11, 1)
i = 0
for x in date:
    current_date = datetime.datetime.strptime(x[:8].decode('ascii'), "%Y%m%d").date()
    day_number = (current_date - INITIAL_DATE).days
    hour = int(x[8:].decode('ascii')) - 1  # start from 0
    is_weekend = 1 if current_date.weekday() > 4 else 0

    dateReadable[i] = [day_number, hour, is_weekend]
    i += 1

date = dateReadable
del dateReadable

# Data cleansing

dayNum, hourNum = 0, 0
isBrokenEntry = False
currentEntryStart, i = 0, 0
dirtyEntries = []
dirtyDays = []

for x in date:
    if x[0] != dayNum:
        if hourNum != 48:
            isBrokenEntry = True
        if isBrokenEntry:
            for j in range(currentEntryStart, i):
                dirtyEntries.append(j)
            dirtyDays.append(dayNum)
        dayNum = x[0]
        hourNum = 0
        currentEntryStart = i
        isBrokenEntry = False
    if x[1] == hourNum:
        hourNum += 1
    else:
        isBrokenEntry = True
        hourNum = x[1] + 1
    i += 1

if hourNum != 48:
    isBrokenEntry = True
if isBrokenEntry:
    for j in range(currentEntryStart, i):
        dirtyEntries.append(j)
    dirtyDays.append(dayNum)

date = np.delete(date, dirtyEntries, axis=0)
data = np.delete(data, dirtyEntries, axis=0)

# Data normalization

i = 0
for x in data:
    j = 0
    for y in x:
        data[i][j] = preprocessing.scale(y)
        j += 1
    i += 1

data = np.reshape(data, (138, 98304))  # only normalized inflow/outflow data in 138 valid days
label = date[::48, 2]  # "1" if this date is in weekend; "0" otherwise

### DATA MODELLING STAGE ###

# SVC classifier
# kernel='rbf'

print("\nSVC classifier (kernel = 'rbf'):\n")

clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
scores = cross_val_score(clf, data, label, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
scores = cross_val_score(clf, data, label, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
scores = cross_val_score(clf, data, label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# kernel='linear'

print("\nSVC classifier (kernel = 'linear'):\n")

clf = svm.SVC(C=1.0, kernel='linear', gamma='auto')
scores = cross_val_score(clf, data, label, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='linear', gamma='auto')
scores = cross_val_score(clf, data, label, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='linear', gamma='auto')
scores = cross_val_score(clf, data, label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# kernel='sigmoid'

print("\nSVC classifier (kernel = 'sigmoid'):\n")

clf = svm.SVC(C=1.0, kernel='sigmoid', gamma='auto')
scores = cross_val_score(clf, data, label, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='sigmoid', gamma='auto')
scores = cross_val_score(clf, data, label, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='sigmoid', gamma='auto')
scores = cross_val_score(clf, data, label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# kernel='poly'

print("\nSVC classifier (kernel = 'poly'):\n")

clf = svm.SVC(C=1.0, kernel='poly', gamma='auto')
scores = cross_val_score(clf, data, label, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='poly', gamma='auto')
scores = cross_val_score(clf, data, label, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(C=1.0, kernel='poly', gamma='auto')
scores = cross_val_score(clf, data, label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# SGD classifier
# loss='hinge'

print("\nSGD classifier (loss='hinge'):\n")

clf = SGDClassifier(loss="hinge", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SGDClassifier(loss="hinge", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SGDClassifier(loss="hinge", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# loss='modified_huber'

print("\nSGD classifier (loss='modified_huber'):\n")

clf = SGDClassifier(loss="modified_huber", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SGDClassifier(loss="modified_huber", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SGDClassifier(loss="modified_huber", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# loss='log'

print("\nSGD classifier (loss='log'):\n")

clf = SGDClassifier(loss="log", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SGDClassifier(loss="log", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = SGDClassifier(loss="log", max_iter=5, tol=-np.infty)
scores = cross_val_score(clf, data, label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 3-neighbor classifier

print("\n3-neighbor classifier:\n")

clf = neighbors.KNeighborsClassifier(n_neighbors=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# 4-neighbor classifier

print("\n4-neighbor classifier:\n")

clf = neighbors.KNeighborsClassifier(n_neighbors=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# 5-neighbor classifier

print("\n5-neighbor classifier:\n")

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# Bagging Classifier (max_samples = 0.5, max_features = 0.5)
# k == 3

print("\nBagging Classifier (max_samples = 0.5, max_features = 0.5, k = 3):\n")

clf = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=3), max_samples=0.5, max_features=0.5)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# k == 4

print("\nBagging Classifier (max_samples = 0.5, max_features = 0.5, k = 4):\n")

clf = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=4), max_samples=0.5, max_features=0.5)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# k == 5

print("\nBagging Classifier (max_samples = 0.5, max_features = 0.5, k = 5):\n")

clf = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=5), max_samples=0.5, max_features=0.5)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# Bagging Classifier (max_samples = 0.5, max_features = 0.2)
# k == 3

print("\nBagging Classifier (max_samples = 0.5, max_features = 0.2, k = 3):\n")

clf = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=3), max_samples=0.5, max_features=0.2)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# k == 4

print("\nBagging Classifier (max_samples = 0.5, max_features = 0.2, k = 4):\n")

clf = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=4), max_samples=0.5, max_features=0.2)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))

# k == 5

print("\nBagging Classifier (max_samples = 0.5, max_features = 0.2, k = 5):\n")

clf = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=5), max_samples=0.5, max_features=0.2)
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=3).mean(), cross_val_score(clf, data, label, cv=3).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=4).mean(), cross_val_score(clf, data, label, cv=4).std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (
    cross_val_score(clf, data, label, cv=5).mean(), cross_val_score(clf, data, label, cv=5).std() * 2))
