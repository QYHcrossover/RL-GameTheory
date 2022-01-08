强化学习有两个基本问题，预测问题和控制问题：

- 预测：给定$MDP(S,A,P,R,y)$以及策略$\pi$,求解该策略下的$V_\pi$
- 控制：给定$MDP(S,A,P, R,y)$,求最优的策略，以及最优策略下的价值函数$V_\pi$

## 预测问题

### MRP

根据上节的贝尔曼方程，可以得到$V_\pi(s)$的递推公式
$$
V_\pi(s) = E[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+...|S_t=s] \\
V_\pi(s) = E[R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+...)|S_t=s] \\
V_\pi(s) = E[R_{t+1}+\gamma V_\pi(S_{t+1})|S_t=s]
$$
进一步化简
$$
V_\pi(s) = E(R_{t+1})+E(V_\pi(S_{t+1})|S_t=s) \\
V_\pi(s) = R_{t+1}+\gamma\sum_{s'\in S}P_{ss'}V_\pi(s')
$$
**矩阵形式:**
$$
V = R + \gamma PV
$$

### MDP

加入决策部分，其递推公式为：

根据
$$
V_\pi(s) = \sum_{a\in A}\pi(a|s)q_\pi(s,a)
$$
而
$$
q_\pi(s,a) = R_{s}^a+\gamma\sum_{s'\in S}P_{ss'}^aV_\pi(s')
$$
所以
$$
V_{k+1}(s) = \sum_{a\in{A}}\pi(a|s)(R_s^a+\gamma\sum_{s'\in{S}}P_{ss'}^aV_{k}(s'))
$$
有了递推公式，就能通过不断迭代各个状态价值函数的值，指导所有的状态函数都稳定，则求出来了。

![image-20211011164506818](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211011164507.png)

## 案例-FrozenLake-v0策略评估

```python
#构造P,r
P = np.zeros((env.unwrapped.nS,env.unwrapped.nS))
r = np.zeros((env.unwrapped.nS,1))
for state in range(env.unwrapped.nS):
    for action in range(env.unwrapped.nA):
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            P[state][next_state] += prob*random_policy[state][action]
            r[state,0] += prob*random_policy[state][action]*reward
```

```python
#迭代法求解价值函数
v = np.zeros((env.unwrapped.nS,1)) #随机初始价值函数
gamma = 1
tolerant=1e-6
it = 0
while True:
    v_new = r + gamma*P@v
    delta = np.max(abs(v_new - v))
    if delta < tolerant:
        break
    it += 1
    v = v_new
    print("第 {} 次迭代，最大差值为 {} ".format(it,delta))
print(it)
print(v_new.reshape(4,4))
```



## 控制问题

### 策略迭代

一种可行的方法就是根据我们之前基于任意一个给定策略评估得到的状态价值来及时调整我们的动作策略，这个方法我们叫做**策略迭代(Policy Iteration)**。

![image-20211012152749215](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211012152750.png)

![img](https://images2018.cnblogs.com/blog/1042406/201808/1042406-20180812191537706-1156414836.jpg)

策略迭代的主要步骤是，首先初始化一个价值函数。根据价值函数调整策略，使用贪婪法调整策略；再进行策略评估得到新的价值函数，根据新的价值函数又调整策略，知道策略稳定没有变化后，则求出了最优策略，和最优策略下的价值函数。

```python
def iterate_policy(env, gamma=1., tolerant=1e-6):
     # 初始化为任意一个策略
    policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) \
            / env.unwrapped.nA
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant) # 策略评估
        if improve_policy(env, v, policy): # 策略改进
            break
    return policy, v
```

### 价值迭代

我们没有等到状态价值收敛才调整策略，而是随着状态价值的迭代及时调整策略, 这样可以大大减少迭代次数。此时我们的状态价值的更新方法也和策略迭代不同。现在的贝尔曼方程迭代式子如下：
$$
V_{k+1}(s) = max(R_s^a+\gamma\sum_{s'\in{S}}V_k(s')) \quad a\in{A}
$$

![image-20211012153038685](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211012153041.png)

```python
def iterate_value(env, gamma=1, tolerant=1e-6):
    #随机初始化一个价值函数，根据迭代公式不断更新价值函数直到价值函数稳定，则为最优策略下的价值函数
    v = np.zeros(env.unwrapped.nS) # 初始化
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vmax = max(v2q(env, v, s, gamma)) # 更新价值函数
            delta = max(delta, abs(v[s]-vmax))
            v[s] = vmax
        if delta < tolerant: # 满足迭代需求
            break
    #根据价值函数反推出最优策略        
    policy = np.zeros((env.unwrapped.nS, env.unwrapped.nA)) # 计算最优策略
    for s in range(env.unwrapped.nS):
        a = np.argmax(v2q(env, v, s, gamma))
        policy[s][a] = 1.
    return policy, v
```

### 策略迭代和价值迭代对比

```python
策略迭代结果:---------------------------------
总迭代次数 = 614
状态价值函数 =
[[0.82351246 0.82350689 0.82350303 0.82350106]
 [0.82351416 0.         0.5294002  0.        ]
 [0.82351683 0.82352026 0.76469786 0.        ]
 [0.         0.88234658 0.94117323 0.        ]]
最优策略 =
[[0 3 3 3]
 [0 0 0 0]
 [3 1 0 0]
 [0 2 1 0]]
```

```python
价值迭代结果:---------------------------------
总迭代次数 = 321
状态价值函数 =
[[0.82351232 0.82350671 0.82350281 0.82350083]
 [0.82351404 0.         0.52940011 0.        ]
 [0.82351673 0.82352018 0.76469779 0.        ]
 [0.         0.88234653 0.94117321 0.        ]]
最优策略 =
[[0 3 3 3]
 [0 0 0 0]
 [3 1 0 0]
 [0 2 1 0]]
```

## 参考资料

- https://github.com/ZhiqingXiao/rl-book/blob/master/chapter03_dp/FrozenLake-v0.ipynb
- https://www.cnblogs.com/pinard/p/9463815.html

