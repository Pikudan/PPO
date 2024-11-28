import sys
import torch
import numpy as np
from runners import EnvRunner


class AsArray:
    """ Converts lists of interactions to ndarray. """
    def __call__(self, trajectory):
        # Modify trajectory inplace.
        for k, v in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[k] = np.asarray(v)


class Policy:
    """ Policy of agent"""
    def __init__(self, model):
        self.model = model

    def act(self, inputs, device='cpu', training=False):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        (mus, sigmas), values = self.model(inputs)
        dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if training:
            return {'distribution': dist,
                    'values': values}
        else:
            return {'actions': actions.detach().numpy(),
                    'log_probs': log_probs.detach().numpy(),
                    'values': values.detach().numpy()}


class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95, device='cpu'):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = device

    def __call__(self, trajectory):
        value_target = self.policy.act(trajectory['state']['latest_observation'], device=self.device)['values'][0]
        env_steps = trajectory['state']['env_steps']
        rewards = torch.tensor(trajectory['rewards'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(trajectory['resets'], dtype=torch.float32, device=self.device)
        is_not_done = 1 - dones
        trajectory['values'] = torch.tensor(trajectory['values'],dtype=torch.float32, device=self.device)
        trajectory['advantages'] = []
        trajectory['value_targets'] = []
        gae = 0
        for step in reversed(range(env_steps)):
            if step==env_steps - 1:
                delta = rewards[step] + self.gamma*value_target*is_not_done[step] - trajectory['values'][step]
            else:
                delta = rewards[step] + self.gamma*trajectory['values'][step + 1]*is_not_done[step] -\
                        trajectory['values'][step]
            
            gae = delta + self.gamma*self.lambda_*is_not_done[step]*gae
            trajectory['advantages'].insert(0, gae)
            trajectory['value_targets'].insert(0, gae + trajectory['values'][step])
        trajectory['advantages'] = torch.tensor(trajectory['advantages'], dtype=torch.float32, device=self.device)
        trajectory['value_targets'] = torch.tensor(trajectory['value_targets'], dtype=torch.float32, device=self.device)


class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """
    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = self.runner.get_next()
        for transform in self.transforms:
                transform(self.trajectory)

    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.
            Should be called at the beginning of each epoch.
        """
        pass

    def get_next(self):
        """ Returns next minibatch.  """
        if self.epoch_count==self.num_epochs:
            self.trajectory = self.runner.get_next()
            for transform in self.transforms:
                transform(self.trajectory)
            self.epoch_count = 0
        minibatch_dict = {}
        rand_inds = np.random.randint(0, self.trajectory['state']['env_steps'], self.num_minibatches)
        for key, value in self.trajectory.items():
            if key!='state':
                if len(value)==2:
                    minibatch_dict[key] = self.trajectory[key][rand_inds,:]
                else:
                    minibatch_dict[key] = self.trajectory[key][rand_inds]
        self.epoch_count += 1
        return minibatch_dict


class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """
    def __call__(self, trajectory):
        adv = trajectory["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        trajectory["advantages"] = adv


def make_ppo_runner(env, policy, num_runner_steps=2048,
                    gamma=0.99, lambda_=0.95,
                    num_epochs=3, num_minibatches=64, device='cpu'):
    """ Creates runner for PPO algorithm. """
    runner_transforms = [AsArray(),
                         GAE(policy, gamma=gamma, lambda_=lambda_, device=device)]

    runner = EnvRunner(env, policy, num_runner_steps,
                       transforms=runner_transforms)

    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(runner, num_epochs=num_epochs,
                                num_minibatches=num_minibatches,
                                transforms=sampler_transforms)
    return sampler


def evaluate(env, agent, n_games=1,  device='cpu', render=False):
    """Plays an a game from start till done, returns per-game rewards """
    agent.train(False)
    game_rewards = []
    done_counter = 0

    if n_games % 10:
        pos_ball = np.repeat(np.linspace(-0.5, 0.5, 10), n_games / 10)
    else:
        pos_ball = np.repeat(np.linspace(-0.5, 0.5, 10), n_games)
 
    for i in range(n_games):
        state, _ = env.reset(pos_ball[i])
        total_reward = 0
        while True:
            if render:
                env.render()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            (mus, sigmas), _ = agent(state)
            dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
            action = dist.sample().cpu().detach().numpy()
            state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            if done or trunc:
                break
        game_rewards.append(total_reward)
    agent.train(True)
    return game_rewards
