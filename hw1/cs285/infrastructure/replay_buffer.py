from cs285.infrastructure.utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        # if self.obs:  # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None:
            self.obs = observations[-self.max_size:]    # [-self.max_size:]截取最后max_size个元素
            self.acs = actions[-self.max_size:]         # 这种切片操作在Python中是安全的
            self.rews = rewards[-self.max_size:]        # 如果超出索引范围会自动调整
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:  # rewards 是 numpy 数组，可以直接 concatenate
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):   # rewards 是列表，需要特殊处理
                    self.rews += rewards    # 列表加法：[np.array([2, 2])] + [np.array([10, 10, 10]), np.array([5, 5])] = [np.array([2, 2]), np.array([10, 10, 10]), np.array([5, 5])]
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

