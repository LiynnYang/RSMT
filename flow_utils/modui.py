# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/9 18:44
# @Author      : gwsun
# @Project     : RSMT
# @File        : devide.py
# @env         : PyCharm
# @Description :

import numpy as np
import os
import ctypes


def callModui(arr, degree):
    dll = ctypes.cdll.LoadLibrary
    path = os.getcwd() + '/shared_so/modui1030.so'
    lib = dll(path)
    arr = arr * 1000000  # 坐标为小数时执行此放缩
    arr_list = arr.tolist()
    x = list(map(round, [arr_list[i][0] for i in range(degree)]))
    y = list(map(round, [arr_list[i][1] for i in range(degree)]))
    edge1 = [0] * degree
    edge2 = [0] * degree
    TenArrType = ctypes.c_int * degree
    ca = TenArrType(*x)
    cb = TenArrType(*y)
    cc = TenArrType(*edge1)
    cd = TenArrType(*edge2)
    lib.modui2.argtypes = (TenArrType, TenArrType, TenArrType, TenArrType,)
    lib.modui2(ca, cb, cc, cd, degree)
    edges = []
    for i in range(degree-1):
        edges.append([cc[i]-1, cd[i]-1])
    return edges


def generate_RMST3(degree, arr):
    edges = callModui(arr, degree)
    return edges


if __name__ == '__main__':
    arr = np.random.rand(1000, 2)
    degree = arr.shape[0]
    edges3 = generate_RMST3(degree, arr)
    print(edges3, sep='\n')


