import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
import xml_env


def get_reward(ob, a, extented_observation, len_pole=0.6, mode="limited_reward_area"):
    """ Get reward. """
    # определем положение конца маятника
    if extented_observation:
        pos_cart = np.sin(ob[1]) * len_pole + ob[0] - ob[4]
    else:
        pos_cart = np.sin(ob[1]) * len_pole + ob[0]

    # получаем награду
    if abs(ob[1]) < 0.2:
        if mode == "limited_reward_area":
            # поощрение в ограниченной области
            reward = np.cos(ob[1]) + np.exp(-np.abs(pos_cart)) * (np.abs(pos_cart) < 0.1)
        elif reward == "unlimited_reward_area":
            # поощрение везде
            reward = np.cos(ob[1]) + np.exp(-np.abs(pos_cart))
        elif mode == "penalty":
            # штраф за отдаление
            reward = np.cos(ob[1]) + 1 - np.abs(pos_cart)
    elif abs(ob[1]) < np.pi / 2:
        # больше нуля при подьеме, меньше нуля при опускании
        reward = np.cos(ob[1]) - max(ob[3] * ob[1], 0.0)
    else:
        reward = np.clip(-ob[1]**2 + 0.1 * ob[3]**2 + 0.001 * a[0]**2 , -np.pi**2, 0.0)
    return reward


class InvertedPendulumEnv(MujocoEnv):
    """ Inverted Pendulum Environment. """
    def __init__(
        self,
        num_observations = 1,
        extented_observation = False,
        target = False,
        mass_use = False,
        upswing = False,
        mass = None,
        test = True
    ):
        self.observation_space=Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space=Box(-3.0, 3.0, (1,), dtype=np.float32)
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.pos_ball = np.zeros(1)
        self.model = mujoco.MjModel.from_xml_string(xml_env.XML_INVERTED_PENDULUM_ENV)
        self.model.opt.timestep = 0.02 / num_observations
        self.num_observations = num_observations
        self.data = mujoco.MjData(self.model)
        self.init_mass = self.model.body_mass.copy()
        self.init_inertia = self.model.body_inertia.copy()
        self.extented_observation = extented_observation
        self.target = target
        self.mass_use = mass_use
        self.upswing = upswing
        self.test = test
        self.mass = mass

        if test:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.reset_model()

    def step(self, action):
        """ Step in environment with action. """
        self.data.ctrl = action
        series_of_observations = []

        # получение следующих состояний и награды
        for _ in range(self.num_observations):
            mujoco.mj_step(self.model, self.data)
            if self.test:
                self.viewer.sync()
            series_of_observations.append(self.obs())

        obs_np = np.concatenate([series_of_observations])
        assert obs_np.shape == (self.num_observations, 4 + self.extented_observation)
        assert series_of_observations[-1].shape == (4 + self.extented_observation, )
        reward = get_reward(series_of_observations[-1], action, self.extented_observation)
        obs_np = obs_np.ravel()

        # определение терминальности
        if self.upswing:
            terminated = bool(not np.isfinite(obs_np).all())
        else:
            terminated = bool(not np.isfinite(obs_np).all() or np.abs(obs_np[-3]) > 0.2)

        # ограничение по времени
        truncated = False
        if self.current_time > 30.0:
            truncated = True
        return obs_np, reward, terminated,  truncated, {}

    def obs(self):
        """ Current observation in game. """
        # приводим угол к [-pi; pi]
        pos = self.data.qpos
        vel = self.data.qvel
        theta = np.mod(pos[1], 2 * np.pi)
        theta = (theta - 2 * np.pi) if theta > np.pi else theta
        pos[1] = theta
        self.data.qpos = pos

        #возвращем состояние
        if self.extented_observation:
            return np.concatenate([pos, vel, self.pos_ball]).ravel()

        relative_pos = np.array([pos[0] - self.pos_ball[0], pos[1]])
        return np.concatenate([relative_pos, vel]).ravel()

    def reset_model(self, pos_ball=None):
        """ Reset environment. """
        # сбрасыаем положения и скорости
        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel

        # меняем положение шарика
        if self.target:
            if pos_ball is not None:
                self.pos_ball = np.array([pos_ball])
            elif not self.test:
                self.pos_ball = np.array([np.random.uniform(-0.5, 0.5)])

        # меняем положение тел
        if self.upswing:
            self.data.qpos[1] = -np.pi
        elif not self.test:
            self.data.qpos[1] = np.random.uniform(-3.0, 3.0)
            self.data.qpos[1] = np.random.uniform(0.2, 0.2)

        # меняем массу и инерцию
        if self.mass_use:
            if self.mass is not None:
                scale_physics = np.repeat(
                    np.array([[1.0], [1.0], [self.mass / self.init_mass[2]]]),
                    3,
                    axis=1
                )
                self.model.body_mass = self.init_mass * scale_physics[:,0]
                self.model.body_inertia = scale_physics * self.init_inertia
            else:
                scale_physics = np.repeat(
                    np.array([[1.0], [1.0], [np.random.uniform(0.1, 10.0)]]),
                    3,
                    axis=1
                )
                self.model.body_mass = self.init_mass * scale_physics[:,0]
                self.model.body_inertia = scale_physics * self.init_inertia

        # сбрасываем время
        self.data.time = 0.0
        self.data.ctrl = 0.0
        series_of_observations = []
        for _ in range(self.num_observations - 1):
            series_of_observations.append(self.obs())
            mujoco.mj_step(self.model, self.data)
        series_of_observations.append(self.obs())
        obs_np = np.concatenate([series_of_observations])
        assert obs_np.shape == (self.num_observations, 4 + self.extented_observation)
        return obs_np.ravel()

    def reset(self, pos_ball=None):
        """ Reset environment. """
        obs = self.reset_model(pos_ball=pos_ball)
        return obs, {}

    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def draw_ball(self, position=[0.0, 0.0, 0.0], color=[1, 0, 0, 1], radius=0.01):
        """ Draw ball. """
        if self.target:
            self.pos_ball = np.array([position[0]])
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=np.array(position),
            mat=np.eye(3).flatten(),
            rgba=np.array(color),
        )
        if self.test:
            self.viewer.user_scn.ngeom = 1

    @property
    def current_time(self):
        """ Get current time. """
        return self.data.time
