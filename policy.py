# -*- encoding: utf-8 -*-
'''
@File    :   policy.py
@Time    :   2022/01/04 18:47:03
@Author  :   QYHcrossover
@Contact :   qinyuheng@zju.edu.cn
'''

from kuhn_poker import Kuhn_Poker
import numpy as np
import copy
'''
策略类，包含以下函数:
    - get_info_sets ： 根据传入的游戏环境，生成信息集
    - action_prob: 输入游戏特定的状态，返回该策略在该状态下的动作概率分布
    - policy_evaluation: 策略评估，给定策略返回当前策略的期望收获
    - best_response: 返回最佳反应策略组，以及max_utility
    - nash_conv: 返回一个策略的nash_conv
'''

class Policy:
    def __init__(self,game): #传入一个游戏环境
        self.game = game
        self.infosets_per_player = self._get_info_sets()
        self.infosets = [j for i in self.infosets_per_player for j in i] #二维list拉平
        self.len_info_sets = len(self.infosets)
        self.len_actions = len(self.game.action_to_str) #这边不是个通用接口
        self.action_probability_array = np.ones([self.len_info_sets,self.len_actions]) / self.len_actions

    def __str__(self):
        return f"len_actions:{self.len_actions}\nlen_info_sets:{self.len_info_sets}\ninfosets_per_player:{self.infosets_per_player}\ninfosets:{self.infosets}\naction_probability_array:\n{self.action_probability_array}\n"

    def _get_info_sets(self):
        infosets_per_player = [[],[]]
        def travel(state):
            nonlocal infosets_per_player
            if state.is_terminal(): return #编历终止条件
            if state.state_to_info() != None:
                info,player_id = state.state_to_info()
                if not info in infosets_per_player[player_id]: #重复的信息集不会添加进去
                    infosets_per_player[player_id].append(info)
            for action in state.legal_actions():
                travel(state.child(action))
        travel(self.game)
        return infosets_per_player

    def action_prob(self,state):
        current_info = state.state_to_info()
        assert current_info !=None and current_info[0] in self.infosets #首先确保该state对应的info是在自己的信息集中出现的
        index = self.infosets.index(current_info[0])
        return self.action_probability_array[index]

    #策略评估,返回期望收获
    def policy_evaluation(self,policy=None,state=None):
        policy = self if policy==None else policy
        state = self.game if state==None else state
        def travel(policy,state):
            if state.is_terminal(): return state.returns()
            if state.is_chance_node():
                return np.sum([prob*travel(policy,state.child(action)) for action,prob in state.chance_outcomes()],axis=0)
            else:
                #在某个玩家的节点上
                action_prob = policy.action_prob(state)
                legal_actions = state.legal_actions()
                return np.sum([prob*travel(policy,state.child(action)) for action,prob in zip(legal_actions,action_prob)],axis=0)
        return travel(policy,state)

    #求出最佳反应策略
    def best_response(self,policy=None,state=None):
        policy = self if policy==None else policy
        state = self.game if state==None else state
        best_response_policy = copy.deepcopy(policy)
        def travel(policy,state,player_id): #计算对于player的最佳反应策略的收获，同时修改best_response_policy
            nonlocal best_response_policy
            if state.is_terminal(): return state.returns()[player_id]
            if state.is_chance_node():
                return np.sum([prob*travel(policy,state.child(action),player_id) for action,prob in state.chance_outcomes()])
            else:
                #在某个玩家的节点上
                action_prob = policy.action_prob(state)
                legal_actions = state.legal_actions()
                if state._current_player != player_id:
                    return np.sum([prob*travel(policy,state.child(action),player_id) for action,prob in zip(legal_actions,action_prob)])
                else:
                    utility = [travel(policy,state.child(action),player_id) for action in legal_actions]
                    best_action = np.argmax(utility)
                    current_info,id = state.state_to_info()
                    index = best_response_policy.infosets.index(current_info)
                    best_response_policy.action_probability_array[index] *= 0
                    best_response_policy.action_probability_array[index][best_action] = 1
                    return max(utility)
        max_utility = [travel(policy,state,player_id) for player_id in range(2)]
        return best_response_policy,np.array(max_utility)

    def nash_conv(self):
        _,max_utility = self.best_response()
        expected_utility = self.policy_evaluation()
        return (max_utility-expected_utility).mean()


if __name__ == "__main__":
    game = Kuhn_Poker()
    policy = Policy(game)

    #检查信息集是否正确输出
    print(policy.infosets_per_player)
    print(policy.infosets)
    print(policy.action_probability_array.shape)
    # #检查特定state的policy策略
    # state = game.child(0)
    # state = state.child(0)
    # print(policy.action_prob(state))
    #检查策略评估的结果
    print(policy.policy_evaluation())
    #检查最佳反应策略的结果
    best_response,max_utility = policy.best_response()
    print(best_response.action_probability_array)
    print(max_utility)
    #检查nash_conv的结果
    print(policy.nash_conv())
                
                