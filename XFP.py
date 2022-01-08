# -*- encoding: utf-8 -*-
'''
@File    :   XFP.py
@Time    :   2022/01/04 18:47:17
@Author  :   QYHcrossover
@Contact :   qinyuheng@zju.edu.cn
'''

from kuhn_poker import Kuhn_Poker
from policy import Policy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)


class XFP:
    def __init__(self,game):
        self.game = game
        self.policy = Policy(game)
        self.epoch = 0

    def solve(self):
        #计算最佳反应策略
        best_response_policy,_ = self.policy.best_response()
        #更新平均策略
        self._update(best_response_policy)
        
    def _update(self,policy):
        lr = 1 / (1 + self.epoch)
        self.policy.action_probability_array *= (1 - lr)
        self.policy.action_probability_array += lr * policy.action_probability_array
        self.epoch += 1

if __name__ == "__main__":
    game = Kuhn_Poker()
    xfp = XFP(game)

    nash_convs = []
    nash_convs.append(xfp.policy.nash_conv())
    for i in tqdm(range(100)):
        #一轮迭代
        xfp.solve()
        #计算每一轮的nash_conv
        nash_convs.append(xfp.policy.nash_conv())
    # print(nash_convs)
    plt.plot(nash_convs)
    plt.show()
    print(xfp.policy.infosets)
    print(xfp.policy.action_probability_array)
    
