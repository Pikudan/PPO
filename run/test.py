import time
import torch
import numpy as np
from network import Network
from envs import InvertedPendulumEnv

def test(
      upswing: bool = False,
      target: bool = False,
      extended_observation: bool = False,
      mass_use: bool = False,
      mass: float = None,
      num_observations: int = 1,
      policy_model: str = None,
      value_model: str = None
):
    '''Тестирование модели'''
    if extended_observation:
        model = Network(shape_in=5 * num_observations, action_shape=1)
    else:
        model = Network(shape_in=4 * num_observations, action_shape=1)

    # Загрузка параметров модели
    model.policy.load_state_dict(torch.load(policy_model))
    model.value.load_state_dict(torch.load(value_model))
    model.train(False)

    # инициализация окружения для игры
    env = InvertedPendulumEnv(
        target=target,
        extented_observation=extended_observation,
        num_observations=num_observations,
        upswing=upswing,
        mass_use=mass_use,
        mass=mass,
        test=True
    )
    s, _ = env.reset()
    target_pos = [0, 0, 0.6]
    last_update = 0
    time.sleep(2.0)
    while env.current_time < 5000:
        if env.current_time - last_update > 5:
            target_pos = [np.random.rand() - 0.5, 0, 0.65]
            env.draw_ball(target_pos, radius=0.01)
            last_update = env.current_time
        s = torch.tensor(s, dtype=torch.float32)
        (mus, sigmas), _ = model(s)
        dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
        action = dist.sample().cpu().detach().numpy()
        s, _, _, _, _ = env.step(action)
        time.sleep(0.01)
