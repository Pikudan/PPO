import numpy as np
import mujoco
import mujoco.viewer
from typing import Any, Callable, Optional, Union
import gymnasium as gym
from gymnasium.spaces import Box

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv

def get_reward(ob, a, extented_observation, len_pole=0.6):
    if extented_observation:
        pos_cart = np.sin(ob[1]) * len_pole + ob[0] - ob[4]
    else:
        pos_cart = np.sin(ob[1]) * len_pole + ob[0]
        
    if abs(ob[1]) < 0.2:
        # поощрение в ограниченной области
        reward = np.cos(ob[1]) + np.exp(-np.abs(pos_cart)) * (np.abs(pos_cart) < 0.1)
        
        # поощрение везде
        # reward = np.cos(ob[1]) + np.exp(-np.abs(pos_cart))
        
        # штраф за отдаление
        # reward = np.cos(ob[1]) + 1 - np.abs(pos_cart)
    elif abs(ob[1]) < np.pi / 2:
        # больше нуля при подьеме, меньше нуля при опускании
        reward = np.cos(ob[1]) - max(ob[3] * ob[1], 0.0)
    else:
        reward = np.clip(-ob[1]**2 + 0.1 * ob[3]**2 + 0.001 * a[0]**2 , -np.pi**2, 0.0)
    return reward

    
class InvertedPendulumEnv(MujocoEnv):
    xml_env = """
    <mujoco model="inverted pendulum">
            <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="160" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        </asset>
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="1" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.0 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-3 3"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
            <!--geom name="ground" type="plane" pos="0 0 0" /-->
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule" group="3"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-100000 100000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
        </actuator>
    </mujoco>
    """

    def __init__(
        self,
        num_observations=1,
        extented_observation=False,
        target=False,
        mass_use=False,
        upswing=False,
        mass=None,
        test=True
    ):
        self.observation_space=Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space=Box(-3.0, 3.0, (1,), dtype=np.float32)
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.pos_ball = np.zeros(1)
        self.model = mujoco.MjModel.from_xml_string(InvertedPendulumEnv.xml_env)
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


    
    def step(self, a):
        self.data.ctrl = a
        series_of_observations = []
        # получение следующих состояний и награды
        for i in range(self.num_observations):
            mujoco.mj_step(self.model, self.data)
            if self.test:
                self.viewer.sync()
            series_of_observations.append(self.obs())
        series_of_observations_np = np.concatenate([series_of_observations])
        assert series_of_observations_np.shape == (self.num_observations, 4 + self.extented_observation)
        assert series_of_observations[-1].shape == (4 + self.extented_observation, )
        reward = get_reward(series_of_observations[-1], a, self.extented_observation)
        series_of_observations_np = series_of_observations_np.ravel()
        # определение терминальности
        if self.upswing:
            terminated = bool(not np.isfinite(series_of_observations_np).all())
        else:
            terminated = bool(not np.isfinite(series_of_observations_np).all()) or bool(np.abs(series_of_observations[-1][1]) > 0.2)
            
        # ограничение по времени
        truncated = False
        if self.current_time > 30.0:
            truncated = True
        
        return series_of_observations_np, reward, terminated,  truncated, {}

    def obs(self):
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
        else:
            relative_pos = np.array([pos[0] - self.pos_ball[0], pos[1]])
            return np.concatenate([relative_pos, vel]).ravel()
            
    def reset_model(self, pos_ball=None):
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
                scale_physics = np.repeat(np.array([[1.0], [1.0], [self.mass / self.init_mass[2]]]), 3, axis=1)
                self.model.body_mass = self.init_mass * scale_physics[:,0]
                self.model.body_inertia = scale_physics * self.init_inertia
            else:
                scale_physics = np.repeat(np.array([[1.0], [1.0], [np.random.uniform(0.1, 10.0)]]), 3, axis=1)
                self.model.body_mass = self.init_mass * scale_physics[:,0]
                self.model.body_inertia = scale_physics * self.init_inertia
           
        # сбрасываем время
        self.data.time = 0.0
        self.data.ctrl = 0.0
        series_of_observations = []
        for i in range(self.num_observations - 1):
            series_of_observations.append(self.obs())
            mujoco.mj_step(self.model, self.data)
        series_of_observations.append(self.obs())
        series_of_observations_np = np.concatenate([series_of_observations])
        assert series_of_observations_np.shape == (self.num_observations, 4 + self.extented_observation)
        return series_of_observations_np.ravel()

    def reset(self, pos_ball=None, seed=None, options=None):
        obs = self.reset_model(pos_ball=pos_ball)
        
        return obs, {}
        
    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def draw_ball(self, position, color=[1, 0, 0, 1], radius=0.01):
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
        return self.data.time
