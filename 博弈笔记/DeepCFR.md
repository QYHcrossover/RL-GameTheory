## XFP vs CFR

![image-20211221204650397](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/image-20211221204650397.png)



## XFP

![image-20211221205349951](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/image-20211221205349951.png)

```
首先初始化一个策略集
循环每个训练轮次:
- 在当前策略集上计算一个最佳反应策略,最好的动作的prob为1，其他动作为0
- 更新平均策略
```

## NFSP

![image-20211221205721985](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/image-20211221205721985.png)

```
首先每个agent有个Q网络，然后有个总的策略网络policy；MRL：存放强化学习四元组(st,at,rt+1,st+1)，MSL：存放最佳策略(St,at)
循环每个训练轮次：
- 根据anticipatory的值，确定best_response模式还是average_response模式
- 如果是best_response模式：
	- 通过agent的Q网络找到Q值最大的action,并将此action的prob置为1；将其他action的prob置为0
    - 将(state,action)送到MSL池子里
- 如果是average_response模式:
	- 通过策略网络输出，各个action的概率；
	- 根据概率分布采样出一个action
	- 将强化学习四元组送到MRL中
- 每隔一定的轮次，训练agent的Q网络和策略网络
```



## CFR

```
循环每个训练轮次:
- 遍历，计算出一次遍历中所有遗憾值
- 通过遗憾匹配算法计算出当前策略
- 更新平均策略
```

```
遍历算法：
- 如果是终止状态，直接返回回报值(一个列表，包含所有玩家的遗憾值)
- 如果是chance_node，则返回回报值的期望 (prob*utility).sum()
- 在某个玩家的轮次上：
  依次算出所有可能的动作后得到的回报值，作为原始回报值 utility
  计算出 回报值的期望 value=(prob*utility).sum()
  计算出 遗憾值 regret = utility - value
  总的遗憾值  +=遗憾值 * 到达该状态的一个概率 
```

代码: [CFR_and_REINFORCE.ipynb](https://github.com/deepmind/open_spiel/blob/master/open_spiel/colabs/CFR_and_REINFORCE.ipynb)

## Deep CFR 

![image-20211218155847455](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/image-20211218155847455.png)

```
首先初始化 优势网络(advantage network)和策略网络(policy network),各个玩家的优势集(adavantage memories),以及总的策略集(strategy memory)
循环每个训练轮次:
	循环每个玩家:
    	循环每个训练轮次中遍历的次数:
    		游戏状态遍历
    	训练每个玩家的优势网络
训练一个总的策略网络
```

![image-20211218160423591](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/image-20211218160423591.png)

```
遍历算法:
- 如果该状态是终止状态，返回回报
- 如果该状态是一个随机节点，那么采样一个动作，返回遍历下个状态后的最终回报
- 如果是当前玩家的轮次，那么：
	- 通过该玩家的优势网络来计算各个动作的遗憾值，通过遗憾匹配算法来生成策略
	- 分别递归地计算采取各个动作后的回报值
	- 通过遗憾匹配生成的策略，以及真实回报值来计算真实的遗憾值
	- 把(当前状态,迭代轮次,以及真实遗憾值)保存在该玩家的优势集中间
- 如果是其他玩家的轮次，那么
	- 还是用该其他玩家的优势函数，计算遗憾值，以及得到该玩家的策略
	- 将(当前状态,迭代轮次,策略)保存在策略集中间
	- 通过采样，选择一个动作，返回遍历下个状态后的最终回报
```

代码: [deep_cfr_pytorch.ipynb](https://github.com/deepmind/open_spiel/blob/master/open_spiel/colabs/deep_cfr_pytorch.ipynb)



