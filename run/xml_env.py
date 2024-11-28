XML_INVERTED_PENDULUM_ENV = """
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
