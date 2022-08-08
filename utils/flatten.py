from typing import OrderedDict
import gym.spaces
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
import numpy as np
from collections import OrderedDict

def build_flattener(space):
    if isinstance(space, gym.spaces.Dict):
        return DictFlattener(space)
    elif isinstance(space, gym.spaces.Box) \
        or isinstance(space, gym.spaces.MultiDiscrete):
        return BoxFlattener(space)
    elif isinstance(space, gym.spaces.Discrete):
        return DiscreteFlattener(space)
    else:
        raise NotImplementedError


class DictFlattener():
    def __init__(self, ori_space):
        self.space = ori_space
        assert isinstance(self.space, gym.spaces.Dict)
        self.size = 0
        self.flattener = OrderedDict()
        for name, space in self.space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                flattener = BoxFlattener(space)
            elif isinstance(space, gym.spaces.Discrete):
                flattener = DiscreteFlattener(space)
            elif isinstance(space, gym.spaces.Dict):
                flattener = DictFlattener(space)

            self.flattener[name] = flattener
            self.size += flattener.size

    def __call__(self, observation):

        assert isinstance(observation, OrderedDict)
        batch = self.get_batch(observation, self)
        if batch == 1:
            array = np.zeros(self.size, )
        else:
            array = np.zeros(self.size)
        self.write(observation, array, 0)
        return array

    def write(self, observation, array, offset):
        for o, f in zip(observation.values(), self.flattener.values()):
            f.write(o, array, offset)
            offset += f.size

    def get_batch(self, observation, flattener):
        if isinstance(observation, dict):
            # 如果是字典的话返回第一个的batch
            for o, f in zip(observation.values(), flattener.flattener.values()):
                return self.get_batch(o, f)
        else:
            return np.asarray(observation).size // flattener.size




class BoxFlattener():
    def __init__(self, ori_space):
        self.space = ori_space
        assert isinstance(self.space, gym.spaces.Box) \
        or isinstance(self.space, gym.spaces.MultiDiscrete)
        self.size = np.product(self.space.shape)

    def __call__(self, observation):
        array = np.array(observation)
        if array.size // self.size == 1:
            return array.ravel()
        else:
            return array.reshape(-1, self.size)
    
    def write(self, observation, array, offset):
        array[..., offset:offset+self.size] = self(observation)
    


class DiscreteFlattener():
    def __init__(self, ori_space):
        self.space = ori_space
        assert isinstance(self.space, gym.spaces.Discrete)
        self.size = 1


    def __call__(self, observation):
        array = np.array(observation, dtype=np.int)
        if array.size == 1:
            return array.item()
        else:
            return array.reshape(-1, 1)

    def write(self, observation, array, offset):
        array[..., offset:offset+1] = self(observation)


if __name__ == "__main__":
    obs_space1 = gym.spaces.MultiDiscrete(([3, 6, 8]))
    flattener = build_flattener(obs_space1)
    obs_space2 = gym.spaces.Box(low=-1, high=1, shape=(3,))
    flattener = build_flattener(obs_space2)
    obs_space3 = gym.spaces.Discrete(5)
    flattener = build_flattener(obs_space3)

    obs_space = gym.spaces.Dict(
        {
            'action1': obs_space2,
            'action2': obs_space3
        }
    )

    flattener = build_flattener(obs_space)

    observation = OrderedDict({'action1': [1.1, 1.2, 1.3], 
    'action2': 2.0
    })

    print(flattener(observation))