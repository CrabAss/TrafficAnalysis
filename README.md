# TrafficAnalysis
Class project of COMP4434.

## Get started

1. Install [JetBrains PyCharm Professional](https://www.jetbrains.com/pycharm/download/). Remember to use Student Free 
License.
2. Install [Anaconda3](https://www.anaconda.com/distribution/#download-section).
3. Open PyCharm, clone this project and open it. Log in to GitHub on PyCharm if necessary.
4. **[IMPORTANT]** Create a folder called `data` and move the dataset files to this folder.
5. Enter `File > Settings > Project: TrafficAnalysis > Project Interpreter`.
6. Click the "gear" button on the top-right corner of the dialog, and then click `Add...`.
7. Select `Conda Environment` on the left, and click `OK` button.
8. Click `OK` button again to close the `Settings` window.
9. Wait until PyCharm is ready. 
10. If a message pops up on the bottom-right corner of the screen which asks you if you want to turn on `Scientific Mode`, 
do not hesitate to turn it on.
11. PyCharm may suggest you to install missing packages. Allow it to do so.
12. You are ready to go.

## Resources

We plan to use `scikit-learn` to implement the part of data model training and testing. Below are some resources for you 
to get familiar with this module.

* [Introduction to scikit-learn](https://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting)
* [User Guide of scikit-learn](https://scikit-learn.org/stable/user_guide.html)
* [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

## Data preprocessing: what has been done

Currently, we do not use weather data to train data models.

### `date`

Originally, `date` is a simple 1d array of strings like `"2015110101"`. After the preprocessing, `date` becomes a 2d 
array  with three columns.

**Column 1**: Since the dates in this dataset starts from 1 Nov 
2015, the strings of date are converted to the number of days since 1 Nov 2015. For example, `"2015111524"` will be 
converted to `14` since `15 - 01 = 14`. 

**Column 2**: The last 2 characters of the string `"2015110101"` means the number of time slot in this day. `"01"` means 
the first time slot which is 00:00 - 00:30. It is converted to integers which start from **zero**. So, `"01"` will be 
converted to `0`.

**Column 3**: If this day is Saturday or Sunday, this value will be `1`. Otherwise, it will be `0`. This is the label 
data for future training and testing.

### `data`

The inflow/outflow data is normalized using `preprocessing.scale()` function from `scikit-learn`.

### Data cleansing

In the given data, we found that some time slots are missing in some of the dates. So we iterate the `date` array to 
figure out those "dirty" dates and remove these data entries.

## Data modelling mechanism (tentative)

We plan to train and test the data model using the whole inflow/outflow data of a single day (with 48 time slots) as a 
single input entry. Therefore, there are 138 valid days (as input entries), and each input entry has 48 * 2 * 32 * 32 
float number elements. It is stored in the `data` array.

There are only 2 possible values of output for each input. `1` means that this day is Saturday or Sunday. `0` means that 
this day is a weekday. It is stored in the `label` array.

Various classification models would be applied.

## Acknowledgement

The dataset used in this experiment is provided in the following paper:

`Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017. `

This paper is available on [arXiv](https://arxiv.org/abs/1610.00081). The dataset is available in [lucktroy/DeepST](https://github.com/lucktroy/DeepST/tree/master/data/TaxiBJ).
