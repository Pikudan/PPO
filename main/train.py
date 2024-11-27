import time
import torch
import numpy as np
from utils import *
from agent import PPO
from network import Network
from writer import create_writer
from torch.utils.tensorboard import SummaryWriter
from envs import InvertedPendulumEnv as InvertedPendulumEnv

def train(
      upswing: bool=False,
      target: bool=False,
      extended_observation: bool=False,
      mass_use: bool=False,
      policy_model: str=None,
      value_model: str=None,
      num_observations: int=1,
      num_epochs: int=16,
      num_runner_steps: int=2048,
      gamma: float=0.99,
      lambda_: float=0.95,
      num_minibatches: int=64
    ):
    '''Обучение модели'''
    
    hparam_dict={
        'num_observations': num_observations,
        'num_epochs': num_epochs,
        'num_runner_steps': num_runner_steps,
        'gamma': gamma,
        'lambda_': lambda_,
        'num_minibatches': num_minibatches,
        'upswing': upswing,
        'target': target,
        'extended_observation': extended_observation,
    }
    
    writer = create_writer(name="experiment", **hparam_dict)
    
    if extended_observation:
        model = Network(shape_in=5 * num_observations, action_shape=1)
    else:
        model = Network(shape_in=4 * num_observations, action_shape=1)
        
    # Загрузка параметров модели
    if policy_model is not None:
        model.policy.load_state_dict(torch.load(policy_model))
    if value_model is not None:
        model.value.load_state_dict(torch.load(value_model))
    policy = Policy(model)
    
    # инициализация окружения для игры
    env = InvertedPendulumEnv(
        num_observations=num_observations,
        target=target,
        extented_observation=extended_observation,
        upswing=upswing,
        mass_use=mass_use,
        test=False
    )
    env2 = InvertedPendulumEnv(
        num_observations=num_observations,
        target=target,
        extented_observation=extended_observation,
        upswing=upswing,
        mass_use=mass_use,
        test=False
    )
    
    runner = make_ppo_runner(
        env=env,
        policy=policy,
        num_epochs=num_epochs,
        num_runner_steps=num_runner_steps,
        gamma=gamma,
        lambda_=lambda_,
        num_minibatches=num_minibatches
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-05)
    ppo = PPO(policy, optimizer)
    
    current_number_of_steps = 0
    MAX_UPDATES = 30_000_000
    best_rewards = np.mean(evaluate(env2, model, n_games=100))
    i = 0
    while current_number_of_steps<=MAX_UPDATES:
        i += 1
        trajectory = runner.get_next()
        ppo.step(trajectory)
        current_number_of_steps = runner.runner.step_var
        if i % 100 == 0:
            rewards = np.mean(evaluate(env2, model, n_games=100))
            print(f"Iter: {i} | "
                  f"{num_epochs} | "
                  f"{num_observations} | "
                  f" value losses: {ppo.values_loss_np:.3f} | "
                  f" policy losses: {ppo.policy_loss_np:.3f} | "
                  f" values: {ppo.values_np:.3f} | "
                  f" grad norms: {ppo.total_norm:.3f} | "
                  f" advantages: {ppo.advantages_np:.3f} | "
                  f" ppo losses: {ppo.ppo_loss_np:.3f} | "
                  f" rewards: {rewards:.3f}"
            )
            if  best_rewards < rewards:
                best_rewards = rewards
                torch.save(policy.model.policy.state_dict(), f'ppo_policy.pth')
                torch.save(policy.model.value.state_dict(), f'ppo_value.pth')
                
            writer.add_scalar("Episode rewards", rewards, i)
            writer.add_scalar("Policy loss", ppo.policy_loss_np, i)
            writer.add_scalar("Value loss", ppo.values_loss_np, i)
            writer.add_scalar("ppo losses", ppo.ppo_loss_np, i)
            writer.add_scalar("Values", ppo.values_np, i)
            writer.add_scalar("grad norms", ppo.total_norm, i)
            writer.add_scalar("advantages", ppo.advantages_np, i)
    writer.close()
    del model
