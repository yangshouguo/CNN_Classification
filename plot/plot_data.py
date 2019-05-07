#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 2:58 PM
# @Author  : 杨寿国
# @mail    : 891584158@qq.com
# @File    : plot_data.py
# @intro:
import matplotlib.pyplot as plt
import numpy as np
data = [
[0, 5.182103157043457,  1],
[110, 4.013250350952148,  1],
[289, 6.327065467834473,  1],
[294, 6.175313949584961,  2],
[485, 4.884316444396973,  1],
# [486, 4.865998268127441,  1],
[487, 4.223080635070801,  1],
[598, 6.556577682495117,  2],
[847, 6.500476837158203,  1],
[862, 4.376387596130371,  3],
[936, 4.011823654174805,  1],
[969, 4.422276973724365,  3],
[1008, 4.093387126922607,  2],
[1009, 4.308513164520264,  2]
]
data = np.array(data)

marker = {1:"*",2:"+",3:">"}
label = {1:"function header", 2:"special instruction sequence", 3:"register using"}
plt.ion()
plt.figure(0)
plt.ylabel('score')
plt.xlabel('instruction position')
plt.title('zsh_exec_O2')

for type in range(1,4):
    mdata = data[data[:,2]==type]

    plt.scatter(mdata[:,0], mdata[:,1], marker=marker[type], label=label[type], c="black")
    for pos,value,flag in zip(mdata[:,0],mdata[:,1],mdata[:,2]):
        # print(inputx[0][pos])
        if (int(pos) == 862):
            plt.annotate("({},{:.2f})".format(hex(int(pos) * 4), value), (pos, value),xytext=(pos-210,value), xycoords='data')
        elif(int(pos) == 1008):
            plt.annotate("({},{:.2f})".format(hex(int(pos) * 4), value), (pos, value),xytext=(pos-200,value), xycoords='data')
        elif(int(pos)==936):
            plt.annotate("({},{:.2f})".format(hex(int(pos) * 4), value), (pos, value),xytext=(pos, value-0.1), xycoords='data'
                         )
        else:
            plt.annotate("({},{:.2f})".format(hex(int(pos) * 4), value), (pos, value),xytext=(pos, value+0.04), xycoords='data'
                     )
        # print("{}:{}, mark=''".format(pos, maxpooled[pos]))

        print(int(pos),hex(int(pos) * 4))
plt.legend()


print('tetetete')