import numpy as np
from collections import namedtuple
from torch.utils.data.sampler import BatchSampler, RandomSampler, SubsetRandomSampler, SequentialSampler
import torch
import time 


class ExperienceReplayBuffer():
    '''
    Experience Replay Buffer
    '''

    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size
        self._transitions = np.zeros(self.buffer_size, dtype=object)
        # the pointer point to the place to be inserted
        self.full_flag = False
        self.data_pointer = 0

    @property
    def current_size(self):
        return self.buffer_size if self.full_flag else self.data_pointer

    @property
    def transitions(self):
        return self._transitions

    @property
    def is_full(self):
        return self.full_flag

    def insert(self, transition):
        '''
        insert the transition to the buffer, new data is set with highest priority
        '''

        self._transitions[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer >= self.buffer_size:
            self.full_flag = True
            self.data_pointer = 0

    def feedforward_generator(self, **kwargs):
        '''
        sample batch data uniformaly in buffer for feedforward network

        Args:
            batch_size: the size of the batch

        Returns:
            batch_transitions: transitions with size (batch_size)
            batch_indices: indices of batch data
        '''

        batch_size = kwargs['batch_size']
        num_batch = kwargs['num_batch']
        batch_sampler = list(BatchSampler(SubsetRandomSampler(range(self.current_size)), batch_size, drop_last=False))

        for i in range(num_batch):
            batch_indices = batch_sampler[i]
            batch_transitions = self._transitions[batch_indices]

            yield batch_transitions, batch_indices
    
    def uniform_generator(self, batch_size):
        batch_indices = np.random.randint(0, self.current_size, size=batch_size)
        batch_transitions = self._transitions[batch_indices]

        return batch_transitions


    def recurrent_generator(self, **kwargs):
        '''
        sample batch data uniformaly in buffer for recurrent neural network

        Args:
            batch_size: the size of the batch
            chunk_length: the length of sequential data
            returns: the returns of s or (s,a)

        Returns:
            batch_transitions: transitions with size (chunk_length * batch_size)
            batch_returns: returns with size (chunk_lengh * batch_size, 1)
            batch_rnn_states_actor: rnn states of actor with size (batch_size)
            batch_rnn_states_critic: rnn states of critic with size (batch_size)
        '''

        batch_size = kwargs['batch_size']
        chunk_length = kwargs['chunk_length']
        returns = kwargs['returns'].reshape(-1)

        # seperate the data to sequential
        dones = np.array([t.done for t in self._transitions]).reshape(-1)
        done_split = np.where(dones == True)[0] + 1
        buffer_split = np.array([0, self.current_size])

        # chunk split is the split index, seperate different episode
        chunk_split = np.unique(np.concatenate([done_split, buffer_split]))
        chunk_split = [np.arange(chunk_split[i], chunk_split[i+1], chunk_length) for i in range(len(chunk_split)-1)] +\
                      [np.array([chunk_split[-1]])]
        chunk_split = np.concatenate(chunk_split, axis=-1)

        # chunk indicies is the start and end index of each chunk
        chunk_indices = list(zip(chunk_split[:-1], chunk_split[1:]))

        num_chunk = len(chunk_indices)

        # sample batch data, recurrent data: [L,N, ...], L is chunk length, N is batch size
        for batch_indices in BatchSampler(SubsetRandomSampler(range(num_chunk)), batch_size, drop_last=False):
            # the szie of last batch might less than given batch size
            L, N = chunk_length, len(batch_indices)
            batch_transitions = np.zeros((L, N), dtype=object)
            batch_returns = np.zeros((L, N))
            # only need the start rnn state in the chunk
            batch_rnn_states_actor = []
            batch_rnn_states_critic = []
            for idx, batch_idx in enumerate(batch_indices):
                start, end = chunk_indices[batch_idx]
                batch_transitions[:end-start,
                                  idx] = self._transitions[start:end]
                batch_returns[:end-start, idx] = returns[start:end]
                batch_rnn_states_actor.append(
                    self._transitions[start].rnn_state_actor)
                batch_rnn_states_critic.append(
                    self._transitions[start].rnn_state_critic)

            # reshape to (L*N)
            batch_transitions = batch_transitions.reshape(L*N)
            batch_returns = batch_returns.reshape(L*N, 1)
            # reshape to (N)
            batch_rnn_states_actor = np.array(batch_rnn_states_actor)
            batch_rnn_states_critic = np.array(batch_rnn_states_critic)

            yield batch_transitions, batch_returns, batch_rnn_states_actor, batch_rnn_states_critic

    def after_update(self):
        '''
        reset the buffer after one update
        '''

        self._transitions = np.zeros(self.buffer_size, dtype=object)
        self.full_flag = False
        self.data_pointer = 0


class PrioritizedExperienceReplayBuffer():
    '''
    Prioritized Experience Replay Buffer
    '''

    def __init__(self, args):
        self.args = args
        # alpha = (Uniform:0, Greedy:1)
        self.alpha = 0.6
        # beta = (PER:0, NotPER:1)
        self.beta = 0.4
        self.epsilon = 0.01
        self.p_upper = 1
        self.per_beta_increase = 1e-3
        self.sum_tree = SumTree(self.args)

    @property
    def current_size(self):
        return len(self.sum_tree.transitions) if self.sum_tree.full_flag else self.sum_tree.data_pointer

    @property
    def transitions(self):
        return self.sum_tree.transitions

    def insert(self, transition):
        '''
        insert the transition to the buffer, new data is set with highest priority
        '''

        max_p = self.sum_tree.max_p

        # the first data is set with highest priority
        if max_p == 0:
            max_p = self.p_upper

        self.sum_tree.add(max_p, transition)

    def prioritized_generator(self, **kwargs):
        '''
        sample batch data with priority in buffer for feedforward network

        Args:
            batch_size: the size of the batch
            num_batch: number of sample batch

        Returns:
            batch_transitions: transitions with size (batch_size)
            batch_tree_idx: tree index of batch transitions
            batch_IS_weight: Importance Sampling weight of batch transitions
        '''

        batch_size = kwargs['batch_size']
        num_batch = kwargs['num_batch']

        for _ in range(num_batch):
            batch_tree_idx = np.zeros(batch_size)
            batch_transitions = np.zeros_like(batch_tree_idx, dtype=object)
            batch_IS_weight = np.zeros_like(batch_tree_idx)

            # update beta
            self.beta = min(self.beta+self.per_beta_increase, 1)
            total_p = self.sum_tree.total_p
            sample_interval = total_p // batch_size

            # minimum probability in sum tree
            min_prob = self.sum_tree.min_p / total_p
            # sample from each interval
            for i in range(batch_size):
                start = i * sample_interval
                end = (i + 1) * sample_interval
                # obtain a random value
                value = np.random.uniform(start, end)
                leaf_idx, p, transition = self.sum_tree.get_leaf(value)
                prob = p / total_p
                # calculate IS_weight by some tricks
                batch_IS_weight[i] = np.power(prob / min_prob, -self.beta)
                batch_tree_idx[i] = leaf_idx
                batch_transitions[i] = transition

            yield batch_transitions, batch_tree_idx.astype(int), batch_IS_weight

    def update_sum_tree(self, batch_tree_idx, td_error: np.ndarray):
        '''
        update the Sum Tree based on new TD error

        Args:
            batch_tree_idx: tree index of batch data
            te_error: r + gammma * Q_tar(s', mu(s')) - Q_eval(s, mu(s)) 
        '''

        p = abs(td_error) + self.epsilon
        p = np.power(np.clip(p, 1e-6, self.p_upper), self.alpha)
        self.sum_tree.batch_update(batch_tree_idx, p)


class SumTree():
    '''
    Store all priority in leaf nodes, each parent nodes is the sum of two children nodes

    Parameter:
        capacity: number of leaf nodes
        trans_idx: index of transition
    '''

    def __init__(self, args):

        self.capacity = args.buffer_size
        self._tree = np.zeros(2*self.capacity-1)
        self.transitions = np.zeros(self.capacity, dtype=object)
        self.data_pointer = 0
        self.depth = int(np.log2(self.capacity)) + 1
        self.full_flag = False

    @property
    def tree(self):
        return self._tree

    def add(self, p, transition):
        '''
        Add new transition to the Sum Tree

        Args:
            p: priority
            transition: (s, a, r, s') 
        '''

        tree_idx = self.capacity - 1 + self.data_pointer
        self.transitions[self.data_pointer] = transition
        self.update(tree_idx, p)

        self.data_pointer += 1

        # cover the old data if exceed the all capacity
        if self.data_pointer >= self.capacity:
            self.full_flag = True
            self.data_pointer = 0

    def update(self, tree_idx, p):
        '''
        update the priority in the tree based single data

        Args:
            tree_idx: index of the tree leaf node
            p: priority
        '''

        # update all parent nodes
        change = p - self._tree[tree_idx]
        self._tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += change

    def batch_update(self, tree_batch_idx, batch_p):
        '''
        update the priorities in the tree based the batch data

        Args:
            tree_batch_idx: batch index of the leaf nodes
            batch_p: batch priorities
        '''

        self._tree[tree_batch_idx] = batch_p

        # some parents nodes index might be overlap
        parent_batch_idx = np.unique((tree_batch_idx - 1) // 2)

        # count the depth of current parent nodes, we update the value of parent nodes by depths
        depth_count = self.depth - 1

        # update priority according to depth
        while depth_count > 0:
            left_batch_idx = parent_batch_idx * 2 + 1
            right_batch_idx = left_batch_idx + 1
            self._tree[parent_batch_idx] = self._tree[left_batch_idx] + \
                self._tree[right_batch_idx]
            parent_batch_idx = np.unique((parent_batch_idx - 1) // 2)
            depth_count -= 1

    def get_leaf(self, v):
        '''
        search for leaf node idx, priority and corresponding transition

        Args:
            v: value

                        0
              1                  2
          3       4         5          6
        7   8   9   10   11   12   13   14

        self.capacity - 1 ~ 2 * self.capacity 
        '''

        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            # parent node is the leaf node, step searching
            if left_idx >= 2 * self.capacity - 1:
                leaf_idx = parent_idx
                break
            else:
                # search for left path
                if v < self._tree[left_idx]:
                    parent_idx = left_idx
                # search for right path
                else:
                    v = v - self._tree[left_idx]
                    parent_idx = right_idx

        data_idx = leaf_idx - self.capacity + 1
        transition = self.transitions[data_idx]
        p = self._tree[leaf_idx]

        return leaf_idx, p, transition

    @property
    def total_p(self):
        '''
        get the total priority in Sum Tree
        '''

        return self._tree[0]

    @property
    def max_p(self):
        '''
        get the maximum priority in Sum Tree 
        '''

        return np.max(self._tree[-self.capacity:])

    @property
    def min_p(self):
        '''
        get the minimum priority in Sum Tree 
        '''

        beg = -self.capacity
        end = self.data_pointer - self.capacity if not self.full_flag else None

        return np.min(self._tree[beg:end])


if __name__ == '__main__':
    buffer = PrioritizedExperienceReplayBuffer()
    T = namedtuple('transition', ['state', 'action', 'reward', 'next_state'])
    # transition = np.zeros(4)
    transition = T(0, 0, 0, 0)
    buffer.insert(transition)
