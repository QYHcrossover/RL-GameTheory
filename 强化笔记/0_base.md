## 强化学习两个实体、三个要素

![image-20211011105237349](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211011105241.png)

两个实体:**环境(environment)**、**智能体(agent)**

三个要素:**状态(state)**、**动作(action)**、**奖励(reward)**

- **状态(state)**: 智能体对当前环境的一种观测
- **动作(action)**：智能体所能采取的行为，动作集是所能采取的所有动作
- **奖励(reward)**：在智能体采取动作后，环境给予的回应

## 马尔科夫决策过程

智能体与环境交互的过程可以用马尔科夫链简单表示出来，这边假设状态的转移仅仅与当前的状态相关，而与上个状态、上上个状态无关。且当前状态下的决策也仅与当前状态相关，与之前的状态无关。

交互过程简单表示成：$S_0,A_0,R_1,S_1,A_1,R2……$,其中:

- 在$S_t$下，根据策略$\pi$，采取$A_t$
- 在$S_t$下以及策略$\pi$的指导下，使用价值函数$V_{\pi}(s)$来表示当前奖励，以及将来会获得的延迟奖励（衰减后的）的和
- 使用$q_{\pi}(s,a)$表示在策略$\pi$和当前状态为s下，采取动作a的价值
- 状态转移，$P_{ss'}$表示从状态s转移到状态s'的概率；而$P_{ss'}^a$表示在状态s下，采取动作a，下个状态为s‘的概率

![image-20211012092107393](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211012092113.png)

![image-20211011111337363](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211011111342.png)

**难点解析，怎样理解$P_{ss'}$与$P_{ss'}^a$的关系？$R_s$与$R_s^a$之间的关系？**

![[公式]](https://www.zhihu.com/equation?tex=P_%7Bss%27%7D%5E%7Ba%7D)是基于![[公式]](https://www.zhihu.com/equation?tex=P_%7Bss%27%7D%5E%7B%5Cpi%7D)的条件概率,根据全概率公式可得，同理也可得收益之间的关系
$$
P_{ss'}= \sum_{a\in{A}}\pi(a|s)P_{ss'}^a
$$

$$
R_s = \sum_{a\in A}\pi(a|s)R_s^a
$$

**怎样理解$V_\pi(s)$需要取期望？**

因为在状态s下转移到状态s‘，这个过程中获得的奖励也是不确定的，所以需要取一个概率分布

## 马尔科夫最优决策

> 最优状态价值函数是所有策略下产生的众多状态价值函数中的最大者

> 同理也可以定义最优动作价值函数是所有策略下产生的众多动作状态价值函数中的最大者

![image-20211011154145072](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211011154147.png)

## 例子

```python
def evaluate_bellman(env, policy, gamma=1.):
    a, b = np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for p, next_state, reward, done in env.P[state][action]:
                a[state, next_state] -= (pi * gamma * p)
                b[state] += (pi * reward * p)
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.P[state][action]:
                q[state][action] += ((reward + gamma * v[next_state]) * p)
    return v, q
```

## 参考文献

- [强化学习（二）马尔科夫决策过程(MDP) - 刘建平Pinard - 博客园 (cnblogs.com)](https://www.cnblogs.com/pinard/p/9426283.html)

- [强化学习笔记（2）——MDP - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/163084195)
- [zhiqingxiao rl-book chapter02_mdp CliffWalking-v0.ipynb](https://github.com/ZhiqingXiao/rl-book/blob/master/chapter02_mdp/CliffWalking-v0.ipynb)

