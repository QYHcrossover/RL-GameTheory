# -*- encoding: utf-8 -*-
'''
@File    :   RM.py
@Time    :   2022/01/04 18:45:48
@Author  :   QYHcrossover
@Contact :   qinyuheng@zju.edu.cn
'''

import numpy as np
import matplotlib.pyplot as plt

class AveragePolicy:
    def __init__(self,policy_len):
        self.policy = np.zeros(policy_len)
        self.step = 0

    def update(self,policy):
        lr = 1 / (1 + self.step)
        self.policy *= (1 - lr)
        self.policy += lr * policy
        self.step += 1


class Player:
    def __init__(self, policy_len, utility, num):
        """
        :param policy_len: 策略个数
        :param utility:  收益矩阵
        :param num: 玩家0写0 玩家1写1
        """
        self.utility = utility
        self.policy_len = policy_len
        self.regrets_sum = np.zeros(self.policy_len)
        self.average_policy = AveragePolicy(self.policy_len)
        self.num = num

    @property
    def policy(self):
        regrets_p = self.regrets_sum.copy()
        regrets_p[regrets_p<0] = 0
        if regrets_p.sum() > 0:
            return regrets_p / regrets_p.sum()
        else:
            return np.ones(self.policy_len) / self.policy_len

    def choose_action(self):
        action = np.random.choice(list(range(self.policy_len)),p=self.policy)
        # print(f"policy:{self.policy}")
        self.average_policy.update(self.policy)
        return action
        # return np.argmax(self.policy)

    def update_regrets(self,this,other):
        if self.num == 0:
            regrets = self.utility[:,other] - self.utility[this,other]
        else:
            regrets = self.utility[other] - self.utility[other,this]
        # print(regrets)
        self.regrets_sum += regrets

    def exploitability(self,other_player):
        op_policy = other_player.average_policy.policy
        # print(f"op_policy:{op_policy}")
        this_policy = self.average_policy.policy
        # print(f"this_policy:{this_policy}")

        op_policy = op_policy.reshape(-1,1) if self.num == 1 else op_policy
        utilitys = np.sum(self.utility * op_policy,axis=1-self.num)
        return np.max(utilitys) - np.sum(this_policy*utilitys)
        


class RM:
    def __init__(self,p0,p1):
        self.p0 = p0
        self.p1 = p1

    def get_nash_equilibrium(self, epoch):
        nash_convs = []
        for i in range(epoch):
            p0a = p0.choose_action()
            p1a = p1.choose_action()
            # print(f"epoch:{i} p0-action:{p0a} p1-action:{p1a}")

            p0.update_regrets(p0a,p1a)
            p1.update_regrets(p1a,p0a)
            # print(f"p0 regrets sum:",self.p0.regrets_sum)
            # print(f"p1 regrets sum:",self.p1.regrets_sum)

            e0 = p0.exploitability(p1)
            e1 = p1.exploitability(p0)
            # print(f"nash conv:{(e0+e1)/2}")
            
            nash_convs.append((e0+e1)/2)
        plt.plot(nash_convs)
        plt.show()
    
    def show_result(self):
        print(f"p0 average policy:",self.p0.average_policy.policy)
        print(f"p1 average policy:",self.p1.average_policy.policy)


if __name__ == "__main__":
    # 以下例子求解如下纳什均衡（囚徒困境）
    #     P0╲P1    坦白    抵赖
    #      坦白   -4，-4   0，-5
    #      抵赖   -5， 0  -1，-1
    
    u1 = np.array(
        [[0, 1, -1],
         [-1, 0, 1],
         [1, -1, 0]]
    )
    u0 = np.array(
        [[0, -1, 1],
         [1, 0, -1],
         [-1, 1, 0]]
    )
    p0 = Player(3, u0, 0)
    p1 = Player(3, u1, 1)

    # # 第一轮p0和p1的各自策略
    # print(p0.policy)
    # print(p1.policy)

    # # 第一轮p0和p1各自选择的动作
    # p0a,p1a = p0.choose_action(),p1.choose_action()
    # print(p0a)
    # print(p1a)

    # # 第一轮完成后遗憾值
    # p0.update_regrets(p0a,p1a)
    # p1.update_regrets(p1a,p0a)

    # 多次迭代
    rm = RM(p0,p1)
    rm.get_nash_equilibrium(1000)
    rm.show_result()
    