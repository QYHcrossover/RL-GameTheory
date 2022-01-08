## 最佳反应

一个玩家(i)相对于其他玩家们(-i)的最佳反应策略可以表示为
$$
\sigma_i^* = br(\sigma_{-i})= argmax_{\sigma_i^{'}\in\sum_i}u_i(\sigma_i^{'},\sigma_{-i})
$$

## 纳什均衡

在策略组𝜎中，如果每个玩家的策略相对于其他玩家的策略而言都是最佳反应策略，那么策略
组𝜎就是一个纳什均衡（Nash equilibrium）,对于 i 
$$
\sigma_i = \sigma_i^* \qquad i\in[1,2,…N]
$$
![image-20211221171415352](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/image-20211221171415352.png)

## $\epsilon$-纳什均衡

![image-20211221165906017](C:\Users\qinyuheng\AppData\Roaming\Typora\typora-user-images\image-20211221165906017.png)

## exploitability可用度

表示某个玩家最佳反应策略的收益 与 当前策略的收益 的 差
$$
\epsilon_i = u(\sigma_i^*,\sigma_{-i})-u(\sigma_i,\sigma_{-i})
$$
**nash_conv:**
$$
nash\_conv = \sum_1^N\epsilon_i
$$

## reference

[知乎|Section 2.2 纳什均衡](https://zhuanlan.zhihu.com/p/409161752)

