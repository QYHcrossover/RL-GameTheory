# -*- encoding: utf-8 -*-
'''
@File    :   my_cfr.py
@Time    :   2022/01/04 18:50:38
@Author  :   QYHcrossover
@Contact :   qinyuheng@zju.edu.cn
'''

from kuhn_poker import Kuhn_Poker
from policy import Policy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import copy

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3) #三位有效数字

def new_reach(reach,pos,prob):
    newr = reach.copy()
    newr[pos] *= prob
    return newr

class CFR:
    def __init__(self,game):
        self.game = game
        self.policy = Policy(game)
        self.policy.infosets = ['1', '1pb', '2', '2pb', '3', '3pb', '2p', '2b', '3p', '3b', '1p', '1b']
        self.curr_policy = copy.deepcopy(self.policy)
        self.regrets = np.zeros_like(self.policy.action_probability_array)
        self.epoch = 0
    
    #递归遍历计算损失
    def _compute_regrets(self,state,reach):
        if state.is_terminal(): return state.returns()
        if state.is_chance_node():
            return np.sum([prob*self._compute_regrets(state.child(action),new_reach(reach,-1,prob)) for action,prob in state.chance_outcomes()],axis=0)
        else:#在一个玩家节点上
            current_player = state._current_player
            action_prob = self.curr_policy.action_prob(state)
            legal_actions = state.legal_actions()
            utility = np.zeros((2,2))
            for action,prob in zip(legal_actions,action_prob):
                utility[action] = self._compute_regrets(state.child(action),new_reach(reach,current_player,prob))
            # utility = np.array([self._compute_regrets(state.child(action),new_reach(reach,current_player,prob)) for action,prob in zip(legal_actions,action_prob)])
            value = np.sum(utility * action_prob.reshape(-1,1),axis=0)
            # value = np.einsum('ap,a->p', utility, action_prob)
            regrets = utility[:,current_player] - value[current_player]
            cfr_prob = np.prod(reach[:current_player])*np.prod(reach[current_player+1:])
            index = self.curr_policy.infosets.index(state.state_to_info()[0])
            self.regrets[index] += regrets*cfr_prob
            return value

    def _regrets_match(self):
        floored_regrets = np.maximum(self.regrets, 1e-16)
        sum_floored_regrets = np.sum(floored_regrets, axis=1, keepdims=True)
        curr_policy = floored_regrets / sum_floored_regrets
        return curr_policy

    def solve(self):
        #计算损失值
        reach = np.ones(2+1) #当前为双人博弈 
        self._compute_regrets(self.game,reach)
        print("regrets:")
        print(self.regrets)
        #遗憾匹配计算出当前策略
        self.curr_policy.action_probability_array = self._regrets_match()
        print("curr_policy")
        print(self.curr_policy.action_probability_array)
        #更新到平均策略上
        lr = 1 / (1 + self.epoch)
        self.policy.action_probability_array *= (1 - lr)
        self.policy.action_probability_array += lr * self.curr_policy.action_probability_array
        self.epoch += 1

if __name__ == "__main__":
    game = Kuhn_Poker()
    cfr = CFR(game)
    print(cfr.policy.infosets)
    nash_convs = []
    cfr.solve()
    cfr.solve()
    # for i in range(100):
    #     cfr.solve()
    #     nash_convs.append(cfr.policy.nash_conv())
    #     if i == 4:
    #         break
    # print(nash_convs[-1])
    # plt.plot(nash_convs)
    # plt.show()
    # print(cfr.policy.infosets)
    # print(cfr.policy.action_probability_array)
