# ！不要修改这里的内容
import os
from _02_Linear_Regression.Linear_Regression import ridge, lasso

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

features = np.array([
    [2.0133330e+03, 1.6400000e+01, 2.8932480e+02, 5.0000000e+00, 2.4982030e+01, 1.2154348e+02],
    [2.0126670e+03, 2.3000000e+01, 1.3099450e+02, 6.0000000e+00, 2.4956630e+01, 1.2153765e+02],
    [2.0131670e+03, 1.9000000e+00, 3.7213860e+02, 7.0000000e+00, 2.4972930e+01, 1.2154026e+02],
    [2.0130000e+03, 5.2000000e+00, 2.4089930e+03, 0.0000000e+00, 2.4955050e+01, 1.2155964e+02],
    [2.0134170e+03, 1.8500000e+01, 2.1757440e+03, 3.0000000e+00, 2.4963300e+01, 1.2151243e+02],
    [2.0130000e+03, 1.3700000e+01, 4.0820150e+03, 0.0000000e+00, 2.4941550e+01, 1.2150381e+02],
    [2.0126670e+03, 5.6000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02],
    [2.0132500e+03, 1.8800000e+01, 3.9096960e+02, 7.0000000e+00, 2.4979230e+01, 1.2153986e+02],
    [2.0130000e+03, 8.1000000e+00, 1.0481010e+02, 5.0000000e+00, 2.4966740e+01, 1.2154067e+02],
    [2.0135000e+03, 6.5000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02]
    ])

labels = np.array([41.2, 37.2, 40.5, 22.3, 28.1, 15.4, 50. , 40.6, 52.5, 63.9])



def test0():
    assert abs((ridge(features[0]) - labels[0])) <= 10


def test1():
    assert abs((ridge(features[1]) - labels[1])) <= 10


def test2():
    assert abs((ridge(features[2]) - labels[2])) <= 10


def test3():
    assert abs((ridge(features[3]) - labels[3])) <= 10


def test4():
    assert abs((ridge(features[4]) - labels[4])) <= 10


def test5():
    assert abs((ridge(features[5]) - labels[5])) <= 10


def test6():
    assert abs((ridge(features[6]) - labels[6])) <= 10


def test7():
    assert abs((ridge(features[7]) - labels[7])) <= 10


def test8():
    assert abs((ridge(features[8]) - labels[8])) <= 10


def test9():
    assert abs((ridge(features[9]) - labels[9])) <= 10


def test10():
    assert abs((lasso(features[0]) - labels[0])) <= 10


def test11():
    assert abs((lasso(features[1]) - labels[1])) <= 10


def test12():
    assert abs((lasso(features[2]) - labels[2])) <= 10


def test13():
    assert abs((lasso(features[3]) - labels[3])) <= 10


def test14():
    assert abs((lasso(features[4]) - labels[4])) <= 10


def test15():
    assert abs((lasso(features[5]) - labels[5])) <= 10


def test16():
    assert abs((lasso(features[6]) - labels[6])) <= 10


def test17():
    assert abs((lasso(features[7]) - labels[7])) <= 10


def test18():
    assert abs((lasso(features[8]) - labels[8])) <= 10


def test19():
    assert abs((lasso(features[9]) - labels[9])) <= 10