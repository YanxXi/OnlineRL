# Online RL

这份代码包括了当前比较流行的online RL baseline 算法代码，并且在3个gym的测试环境上跑通并且通过调试以及加入GAE(online) 或 PER (offline) 可以取得state-of-art的效果。

## ENV

- 'CartPole-v1' -4个状态，一个二维的离散动作

- 'LunarLanderContinuous-v2' -8个状态，一个连续动作

- 'LunarLander-v2' - 8个状态，一个四维离散动作

- 'BipedalWalker-v3'- 24观测值，4个连续动作

  需要对原环境进行封装，将reward clip 到（-1, 1）

## ALGORITHM

- dqn
- double dqn
- dueling dqn
- a3c
- ppo 
- ddpg
- td3
- sac



## Command

### PPO

####  'CartPole-v0' 

 **100 个 episode 内求解** 

```python
# train ppo
python train_rl.py --env CartPole-v0 ppo --use-recurrent-layer --save-model
# test ppo
python test_rl.py --env CartPole-v0 --save-time 01_22_11_02 ppo --use-recurrent-layer
# collect data
python collect_data.py --env CartPole-v0 --target-label action --save-time 01_22_11_02 ppo --use-recurrent-layer
```

#### 'LunarLander-v2'

**300 个 episode 内求解**

```python
# train ppo
python train_rl.py --env LunarLander-v2 ppo --use-recurrent-layer --save-model
# test ppo
python test_rl.py --env LunarLander-v2 --save-time 01_21_21_12 ppo --use-recurrent-layer
# collect data
python collect_data.py --env LunarLander-v2 --target-label action --save-time 01_21_21_12 ppo --use-recurrent-layer
```



### DDPG

####  'LunarLanderContinuous-v2'

**300 个episode 内求解**

```python
# train ddpg
python train_rl.py --env LunarLanderContinuous-v2 ddpg --use-per --use-batch-normalization --eval-in-paral --save-model
# test ddpg
python test_rl.py --env LunarLanderContinuous-v2 --save-time 01_22_11_16 ddpg --use-batch-normalization
```



### TD3

####  'BipedalWalker-v3'

**70 个epoch 内求解**

```python
# train td3
python train_rl.py --env BipedalWalker-v3 td3 --steps-per-epoch 5000 --max-episode-steps 1000 --save-model
# test td3
python test_rl.py --env BipedalWalker-v3 --save-time 05_16_11_55 td3
```



### SAC

####  'CartPole-v1'

**30 个epoch 内求解**

```python
# train sac
python train_rl.py --env CartPole-v1 sac --steps-per-epoch 500 --save-model
# test sac
python test_rl.py --env CartPole-v1 --save-time 05_12_09_54 sac
```



####  'BipedalWalker-v3'

**60 个epoch 内求解**

```python
# train sac
python train_rl.py --env BipedalWalker-v3 sac --steps-per-epoch 5000 --max-episode-steps 1000 --save-model
# test sac
python test_rl.py --env BipedalWalker-v3 --save-time 05_16_18_10 sac
```



### DQN

####  'CartPole-v1'

 **20 个 epoch 内求解** 

```python
# train dqn 
python train_rl.py --env CartPole-v1 dqn --steps-per-epoch 500 --save-model
# test dqn
python test_rl.py --env CartPole-v1 --save-time 05_16_11_01 dqn
```



### Double DQN

####  'CartPole-v1'

 **20 个 epoch 内求解(可能是因为我并没有去调整double dqn的参数，参数同dqn相同)** 

```python
# train double dqn
python train_rl.py --env CartPole-v1 dōubledqn --steps-per-epoch 500 --save-model
# test double dqn
python test_rl.py --env CartPole-v1 --save-time 05_16_11_05 doubledqn
```



### Dueling DQN

####  'CartPole-v1'

 **20个 epoch 内求解(但是很不稳定，有时候很快能求解有时一直不收敛，参数同dqn相同)** 

```python
# train dueling dqn
python train_rl.py --env CartPole-v1 duelingdqn --steps-per-epoch 500 --save-model
# test dueling dqn
python test_rl.py --env CartPole-v1 --save-time 05_16_11_28 duelingdqn
```



### A3C

实现了一个版本但没有进行训练
