import h5py  # h5py package is a Pythonic interface to the HDF5 binary data format.
# http://docs.h5py.org/en/stable/quick.html.
import numpy  # numPy is the fundamental package for scientific computing with Python.
# https://docs.scipy.org/doc/numpy-1.16.1/user/quickstart.html.

from sklearn import preprocessing


# Read BJ16_M32x32_T30_InOut.h5.

f1 = h5py.File("data/BJ16_M32x32_T30_InOut.h5", "r")
print("Subsets in BJ16_M32x32_T30_InOut.h5:")
# List names and shapes of all datasets in the file.
for key in f1.keys():
    print(key, f1[key].shape)
# the following lines will be shown on the screen:
# data (7220, 2, 32, 32)
# date (7220,)

# Read BJ16_M32x32_T30_InOut.h5.

f2 = h5py.File("data/BJ_Meteorology.h5", "r")
print("Subsets in BJ_Meteorology.h5:")
# List names and shapes of all datasets in the file.
for key in f2.keys():
    print(key, f2[key].shape)
# the following lines will be shown on the screen:
# Temperature (7220,)
# Weather (7220, 17)
# WindSpeed (7220,)
# date (7220,)


# Transform a datasets in a .h5 file into an numpy array.

date = f1["date"][:]
# A array of shape 7220x1 where date[i] is the time of i-th timeslot.




data = f1["data"][:]
# data_scaled = preprocessing.scale(data)
# An array of shape 7220x2x32x32 where the first dimension represents the index of timeslots. data[i][0] is a (32, 32) inflow matrix and data[i][1] is a (32, 32) outflow matrix.

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
