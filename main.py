import datetime

import h5py
import numpy as np
from sklearn import preprocessing
from sklearn import svm

### DATA READING STAGE ###

f1 = h5py.File("data/BJ16_M32x32_T30_InOut.h5", "r")
f2 = h5py.File("data/BJ_Meteorology.h5", "r")

# Transform a datasets in a .h5 file into an numpy array.

date = f1["date"][:]
# A array of shape 7220x1 where date[i] is the time of i-th timeslot.

data = f1["data"][:]
# An array of shape 7220x2x32x32 where the first dimension represents the index of timeslots. data[i][0] is a (32,
# 32) inflow matrix and data[i][1] is a (32, 32) outflow matrix.

temperature = f2["Temperature"][:]
# An array of shape 7220x1 where temperature[i] is the temperature at i-th timeslot.

weather = f2["Weather"][:]
# An array of shape 7220x17 where weather[i] is a one-hot vector which means:
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Sunny
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Cloudy
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Overcast
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Rainy
# [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Sprinkle
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ModerateRain
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] HeavyRain
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] Rainstorm
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] Thunderstorm
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] FreezingRain
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] Snowy
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] LightSnow
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] ModerateSnow
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] HeavySnow
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] Foggy
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] Sandstorm
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] Dusty

windspeed = f2["WindSpeed"][:]
# An array of shape 7220x1 where windspeed[i] is the wind speed at i-th timeslot.


f1.close()
f2.close()

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

print("Number of valid entries:" + str(len(date) - len(dirtyEntries)))

date = np.delete(date, dirtyEntries, axis=0)
data = np.delete(data, dirtyEntries, axis=0)
# temperature = np.delete(temperature, dirtyEntries, axis=0)
# weather = np.delete(weather, dirtyEntries, axis=0)
# windspeed = np.delete(windspeed, dirtyEntries, axis=0)


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

# Please refer to the link below to learn more:
# Introduction: https://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting
# User Guide: https://scikit-learn.org/stable/user_guide.html
# Choosing the right estimator: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# The code below is only an example. This model performs VERY POOR.
# Need improvement.

clf = svm.SVC(gamma=0.001, C=100.)  # Parameter tuning NEEDED.

clf.fit(data[:-10], label[:-10])
print(clf.predict(data[-10:]))

# ACTUAL RESULT: [0 0 1 1 0 0 0 0 0 1]
