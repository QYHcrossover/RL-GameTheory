# -*- encoding: utf-8 -*-
'''
@File    :   kuhn_poker.py
@Time    :   2022/01/04 18:46:19
@Author  :   QYHcrossover
@Contact :   qinyuheng@zju.edu.cn
'''

from copy import deepcopy
from itertools import chain
import numpy as np

def add_str(str1,str2):
    new = ""
    for i in range(max(len(str1),len(str2))):
        if i < len(str1):
            new += str1[i]
        if i < len(str2):
            new += str2[i]
    return new

class Kuhn_Poker:
    def __init__(self):
        #定义博弈树 
        u1_ = {"p":{"p":np.array([-1,1]),"b":{"p":np.array([-1,1]),"b":np.array([-2,2])}},"b":{"p":np.array([1,-1]),"b":np.array([-2,2])}}
        u21 = {"p":{"p":np.array([1,-1]),"b":{"p":np.array([-1,1]),"b":np.array([2,-2])}},"b":{"p":np.array([1,-1]),"b":np.array([2,-2])}}
        u23 = {"p":{"p":np.array([-1,1]),"b":{"p":np.array([-1,1]),"b":np.array([-2,2])}},"b":{"p":np.array([1,-1]),"b":np.array([-2,2])}}    
        u3_ = {"p":{"p":np.array([1,-1]),"b":{"p":np.array([-1,1]),"b":np.array([2,-2])}},"b":{"p":np.array([1,-1]),"b":np.array([2,-2])}}    
        self._uility_tree = {"12":u1_,"13":u1_,"21":u21,"23":u23,"31":u3_,"32":u3_}
        self.action_to_str = ["p","b"]
        self.str_to_action = {s:i for i,s in enumerate(self.action_to_str)}
        self.new_init_state()
    
    def new_init_state(self):
        self._have = "" #起初两个玩家分到的牌
        self._history = ["",""] #两个玩家执行的动作序列
        self._current_tree = self._uility_tree.copy() #当前的博弈树
        self._current_player = 0
        return self

    def is_chance_node(self): #发牌阶段为chance_node
        return len(self._have) < 2

    def is_terminal(self): #是否为终止节点
        return type(self._current_tree) == np.ndarray
    
    def returns(self):
        assert self.is_terminal()
        return self._current_tree

    def legal_actions(self): #返回合法动作  
        if self.is_chance_node():
            return list(range(3-len(self._have)))
        else:
            return list(range(2))
    
    #仅在chance_node中使用，返回 [(action,prob),]
    def chance_outcomes(self):
        assert self.is_chance_node()
        return zip(range(3-len(self._have)),[1/(3-len(self._have))]*(3-len(self._have)))
    
    def child(self,action):
        assert action in self.legal_actions()
        state = deepcopy(self) #deepcopy很重要
        if state.is_chance_node():
            left = sorted(list(set("123") - set(state._have)))
            state._have += left[action]
            if not state.is_chance_node(): #发牌阶段完成
                state._current_tree = state._current_tree[state._have]
        else:
            action_str = state.action_to_str[action] #p或者b
            state._history[state._current_player] += action_str
            state._current_tree = state._current_tree[action_str]
            state._current_player = abs(state._current_player-1)
        return state

    def state_to_info(self):
        if len(self._have) < 2 or self.is_terminal(): return None #发牌阶段以及终止阶段不算信息集
        return self._have[0] + add_str(*self._history),self._current_player #总是player1获得的牌,player1和player2的行动序列
        
if __name__ == "__main__":
    state = Kuhn_Poker()
    def travel(state):
        if state.is_terminal(): return state.returns()
        utility_tree = {}
        for action in state.legal_actions():
            utility_tree[action] = travel(state.child(action))
        return utility_tree
    utility_tree = travel(state)
    print(utility_tree[0][0]) #u12
    print(utility_tree[0][1]) #u13
    print(utility_tree[1][0]) #u21
    print(utility_tree[1][1]) #u23
    print(utility_tree[2][0]) #u31
    print(utility_tree[2][1]) #u32

    