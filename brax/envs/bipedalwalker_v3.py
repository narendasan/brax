from typing import Optional, Tuple, Dict

import brax
from brax import jumpy as jp
from brax import math
from brax.envs import env
from google.protobuf import text_format

from brax.io import image

import numpy as np
import jax
from jax import numpy as jnp

import gym
from gym import spaces

# TODO: stable physics / working heuristic
# TODO: jit env.step?
# TODO: hardcore mode
# TODO: normalized rewards
# TODO: rendering cosmetics

VIEWPORT_W = 600
VIEWPORT_H = 400

SCALE = 15  # pixels per meter

GROUND, HULL, LEFT_THIGH, RIGHT_THIGH, LEFT_LEG, RIGHT_LEG = range(6)
LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE = range(4)

HULL_MASS = 42.0
LIMB_MASS = 1.5

HULL_X = 1.2
HULL_Z = 0.4

HIP_OFFSET_X = 0.1
THIGH_X = 0.3
THIGH_Z = 1.2

LEG_X = THIGH_X * 0.8
LEG_Z = THIGH_Z

STRENGTH_HIP = 50
STRENGTH_KNEE = STRENGTH_HIP * 4 / 6

TERRAIN_STARTPAD = 60  # in steps
TERRAIN_LENGTH = 240  # in steps
TERRAIN_SCALE = 10
TERRAIN_SIZE = TERRAIN_LENGTH / 2
INITIAL_X = 40 / TERRAIN_LENGTH * TERRAIN_SIZE

LIDAR_RANGE = 10.0


class BipedalWalker(env.Env):
    """
    ### Description
    This is a simple 4-joint walker robot environment.
    There are two versions:
    - Normal, with slightly uneven terrain.
    - Hardcore, with ladders, stumps, pitfalls.

    To solve the normal version, you need to get 300 points in 1600 time steps.
    To solve the hardcore version, you need 300 points in 2000 time steps.

    A heuristic is provided for testing. It's also useful to get demonstrations
    to learn from. To run the heuristic:
    ```
    python gym/envs/box2d/bipedal_walker.py
    ```

    ### Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.

    ### Observation Space
    State consists of hull angle speed, angular velocity, horizontal speed,
    vertical speed, position of joints and joints angular speed, legs contact
    with ground, and 10 lidar rangefinder measurements. There are no coordinates
    in the state vector.

    ### Rewards
    Reward is given for moving forward, totaling 300+ points up to the far end.
    If the robot falls, it gets -100. Applying motor torque costs a small
    amount of points. A more optimal agent will get a better score.

    ### Starting State
    The walker starts standing at the left end of the terrain with the hull
    horizontal, and both legs in the same position with a slight knee angle.

    ### Episode Termination
    The episode will terminate if the hull gets in contact with the ground or
    if the walker exceeds the right end of the terrain length.

    ### Arguments
    To use to the _hardcore_ environment, you need to specify the
    `hardcore=True` argument like below:
    ```python
    import gym
    env = gym.make("BipedalWalker-v4", hardcore=True)
    ```

    ### Version History
    - v4: Replaced box2d with brax
    - v3: returns closest lidar trace instead of furthest;
        faster video recording
    - v2: Count energy spent
    - v1: Legs now report contact with ground; motors have higher torque and
        speed; ground has higher friction; lidar rendered less nervously.
    - v0: Initial version


    <!-- ### References -->

    ### Credits
    Created by Oleg Klimov

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 100,
    }

    def __init__(self,
                hardcore: bool = False,
                **kwargs):
        hardcore = hardcore
        terrain_x, terrain_y, terrain_points, terrain_map = self._generate_terrain(jp.random_prngkey(seed=0), hardcore)
        config = _SYSTEM_CONFIG(terrain_map)
        with open('sys_conf.pbtxt', 'w') as f:
            f.write(config)
        super().__init__(config=config, **kwargs)

        self.hardcore = hardcore

        # we use 5.0 to represent the joints moving at maximum
        # 5 x the rated speed due to impulses from ground contact etc.
        low = np.array(
            [
                -jp.pi,
                -5.0,
                -5.0,
                -5.0,
                -jp.pi,
                -5.0,
                -jp.pi,
                -5.0,
                -0.0,
                -jp.pi,
                -5.0,
                -jp.pi,
                -5.0,
                -0.0,
            ]
            + [-1.0] * 10
        ).astype(np.float32)
        high = np.array(
            [
                jp.pi,
                5.0,
                5.0,
                5.0,
                jp.pi,
                5.0,
                jp.pi,
                5.0,
                5.0,
                jp.pi,
                5.0,
                jp.pi,
                5.0,
                5.0,
            ]
            + [1.0] * 10
        ).astype(np.float32)
        self.action_space = spaces.Box(
            jp.array([-1, -1, -1, -1]).astype(np.float32),
            jp.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)


    def _to_2d(self, arr: jp.ndarray) -> jp.ndarray:
        return arr[jp.array([0, 2])]

    def _get_angle(self, rot) :
        # returns angle around y axis from rotation quaternion
        return 2 * jp.arctan2(rot[2], rot[0])

    def _get_2d_box_size(self, body):
        body_halfsizes = body.colliders[0].box.halfsize
        return 2 * jp.array([body_halfsizes.x, body_halfsizes.z])

    def _get_lidar(self, qp, env_info):
        hull_pos = self._to_2d(qp.pos[HULL])
        directions = jp.array(
            [[jp.sin(1.5 * i / 10.0), -jp.cos(1.5 * i / 10.0)] for i in range(10)]
        )
        p1 = env_info["terrain_points"][:-1]
        p2 = env_info["terrain_points"][1:]
        v1 = hull_pos - p1
        v2 = p2 - p1
        v3 = jp.array([-directions.T[1], directions.T[0]]).T
        distances = jp.cross(v2, v1)[:, None] / (v2 @ v3.T)
        intersect_points = hull_pos + jnp.einsum("ij,ki->ikj", directions, distances)
        valid_intersects = jp.array(
            [
                (
                    (env_info["terrain_points"][:-1, 0] <= intersect_points[i][:, 0])
                    & (intersect_points[i][:, 0] <= env_info["terrain_points"][1:, 0])
                )
                for i in range(len(directions))
            ]
        ).T
        min_distances = jp.minimum(
            LIDAR_RANGE, jp.where(valid_intersects, distances, LIDAR_RANGE)
        ).min(axis=0)
        normed_min_distances = min_distances / LIDAR_RANGE
        return normed_min_distances

    def _get_obs(self, qp, env_info) -> Tuple[jp.ndarray, Dict]:
        left_leg_ground_contact, right_leg_ground_contact = (
            env_info["system_info"].contact.vel[(LEFT_LEG, RIGHT_LEG), 2] != 0
        ).astype(int)
        joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)
        joint_angle += jp.array([0, 1, 0, 1]) # SHOULD THIS BE APPEND OR ADD?
        joint_vel /= jp.array([6, 4, 6, 4])

        hull_velocity = self._to_2d(0.3 * qp.vel[HULL])

        lidar_detections = self._get_lidar(qp, env_info)

        obs = jnp.array([
            self._get_angle(qp.rot[HULL]),
            qp.ang[HULL][1] / 200,
            *self._to_2d(0.3 * qp.vel[HULL]),
            joint_angle[LEFT_HIP],
            joint_vel[LEFT_HIP],
            joint_angle[LEFT_KNEE],
            joint_vel[LEFT_KNEE],
            left_leg_ground_contact,
            joint_angle[RIGHT_HIP],
            joint_vel[RIGHT_HIP],
            joint_angle[RIGHT_KNEE],
            joint_vel[RIGHT_KNEE],
            right_leg_ground_contact,
            *lidar_detections,
        ])

        metrics = {
            "hull_rot": self._get_angle(qp.rot[HULL]),
            "hull_something_else": qp.ang[HULL][1] / 200,
            "hull_x_velocity": hull_velocity[0], # Note sure if it is xy or yx
            "hull_y_velocity": hull_velocity[1], # Note sure if it is xy or yx
            "left_hip_joint_angle": joint_angle[LEFT_HIP],
            "left_hip_joint_velocity": joint_vel[LEFT_HIP],
            "left_knee_joint_angle": joint_angle[LEFT_KNEE],
            "left_knee_joint_velocity": joint_vel[LEFT_KNEE],
            "left_leg_ground_contact": left_leg_ground_contact,
            "right_hip_joint_angle": joint_angle[RIGHT_HIP],
            "right_hip_joint_velocity": joint_vel[RIGHT_HIP],
            "right_knee_joint_angle": joint_angle[RIGHT_KNEE],
            "right_knee_joint_velocity": joint_vel[RIGHT_KNEE],
            "right_leg_ground_contact": right_leg_ground_contact,
            "lidar_intersects": lidar_detections,
        }
        assert len(obs) == 24
        print(type(obs), type(metrics))
        return obs, metrics

    def _generate_terrain(self, rng: jp.ndarray,  hardcore: bool):
        terrain_x = jnp.linspace(
            -INITIAL_X, TERRAIN_SIZE - INITIAL_X, TERRAIN_LENGTH
        )
        velocity = 0
        y = 0
        terrain_y = [0] * TERRAIN_STARTPAD

        for _ in range(TERRAIN_LENGTH - TERRAIN_STARTPAD):
            velocity += (
                -0.2 * velocity
                - 0.01 * jp.sign(y)
                + jp.random_uniform(rng=rng, low=-1, high=1) / TERRAIN_SCALE
            )
            y += velocity
            terrain_y.append(y)

        terrain_y = jp.array(terrain_y)
        terrain_points = jp.array([terrain_x, terrain_y]).T
        terrain_map = jp.repeat(terrain_y, TERRAIN_LENGTH).tolist()
        return (terrain_x, terrain_y, terrain_points, terrain_map)

    def reset(self, rng: jp.ndarray) -> env.State:
        self.prev_shaping = None
        terrain_x, terrain_y, terrain_points, terrain_map = self._generate_terrain(rng, self.hardcore)
        new_sys_conf = _SYSTEM_CONFIG(terrain_map=terrain_map)

        env_info = {
            "terrain_x": terrain_x,
            "terrain_y": terrain_y,
            "terrain_points": terrain_points,
            "terrain_map": terrain_map
        }

        print("Generating a new enviornment map")
        self.sys = brax.System(text_format.Parse(new_sys_conf, brax.Config()))

        self.body_sizes = [
            self._get_2d_box_size(body) for body in self.sys.config.bodies
        ]

        qp = self.sys.default_qp()
        info = self.sys.info(qp)

        env_info["system_info"] = info

        obs, metrics = self._get_obs(qp, env_info)
        reward, done, zero = jp.zeros(3)

        env_state = env.State(
            qp,
            obs,
            reward,
            done,
            metrics,
            env_info
        )

        return env_state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        prev_env_state = state
        env_info = prev_env_state.info

        action = jp.clip(action, -1, +1).astype(jp.float32)
        qp, info = self.sys.step(prev_env_state.qp, action)
        env_info.update(system_info=info)

        obs, metrics = self._get_obs(qp, env_info)

        shaping = (
            3.5 * qp.pos[HULL][0]
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(
            prev_env_state.obs[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        done = False
        #if qp.pos[HULL][0] < -INITIAL_X / 2 or info.contact.vel[HULL, 2] != 0:
        #    reward = -100
        #    done = True
        #if qp.pos[HULL][0] > TERRAIN_SIZE - INITIAL_X:
        #    done = True

        for a in action:
            reward -= 0.028 * jp.clip(jp.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        return state.replace(
            qp=qp,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=env_info,
        )

_SYSTEM_CONFIG = lambda terrain_map:  """
        dt: 0.02
        substeps: 20
        friction: 1.0
        dynamics_mode: "pbd"
        gravity { z: -10 }
        velocity_damping: 1.0
        bodies {
            name: "ground"
            colliders {
                heightMap {
                    size: %s
                    data: %s
                }
            }
            frozen { all: true }
        }
        bodies {
            name: "hull"
            colliders { box { halfsize { x: %s y: 0.5 z: %s }}}
            inertia { x: 1.0 y: 1000.0 z: 1.0 }
            mass: %s
        }
        bodies {
            name: "left_thigh"
            colliders {
            box {
                halfsize { x: %s y: 0.2 z: %s }
            }
            }
            inertia { x: 1.0 y: 1.0 z: 1.0 }
            mass: %s
        }
        bodies {
            name: "right_thigh"
            colliders {
            box {
                halfsize { x: %s y: 0.2 z: %s }
            }
            }
            inertia { x: 1.0 y: 1.0 z: 1.0 }
            mass: %s
        }
        bodies {
            name: "left_leg"
            colliders {
            box {
                halfsize { x: %s y: 0.2 z: %s }
            }
            }
            inertia { x: 1.0 y: 1.0 z: 1.0 }
            mass: %s
        }
        bodies {
            name: "right_leg"
            colliders {
            box {
                halfsize { x: %s y: 0.2 z: %s }
            }
            }
            inertia { x: 1.0 y: 1.0 z: 1.0 }
            mass: %s
        }
        joints {
            name: "left_hip"
            parent: "hull"
            child: "left_thigh"
            parent_offset { x: %s z: -%s }
            child_offset { x: %s z: %s }
            rotation { z: -90.0 y: 0.0 }
            angle_limit { min: -46.0 max: 63.0 }
            angular_damping: 1.0
        }
        joints {
            name: "right_hip"
            parent: "hull"
            child: "right_thigh"
            parent_offset { x: %s z: -%s }
            child_offset { x: %s z: %s }
            rotation { z: -90.0 y: 0.0 }
            angle_limit { min: -46.0 max: 63.0 }
            angular_damping: 1.0
        }
        joints {
            name: "left_knee"
            parent: "left_thigh"
            child: "left_leg"
            parent_offset { z: -%s }
            child_offset { z: %s }
            rotation { z: -90.0 y: 0.0 }
            angle_limit { min: -91.0 max: -5.0 }
            angular_damping: 1.0
        }
        joints {
            name: "right_knee"
            parent: "right_thigh"
            child: "right_leg"
            parent_offset { z: -%s }
            child_offset { z: %s }
            rotation { z: -90.0 y: 0.0 }
            angle_limit { min: -91.0 max: -5.0 }
            angular_damping: 1.0
        }
        actuators {
            name: "left_hip"
            joint: "left_hip"
            strength: %s
            torque {}
        }
        actuators {
            name: "left_knee"
            joint: "left_knee"
            strength: %s
            torque {}
        }
        actuators {
            name: "right_hip"
            joint: "right_hip"
            strength: %s
            torque {}
        }
        actuators {
            name: "right_knee"
            joint: "right_knee"
            strength: %s
            torque {}
        }
        frozen {
            position { y: 1.0 }
            rotation { x: 1.0 z: 1.0 }
        }
        defaults {
            qps { name: "ground" pos { x: -%s y: %s z: 0 }}
            angles { name: "left_hip" angle { x: -20.0 } }
            angles { name: "right_hip" angle { x: 25.0 } }
            angles { name: "left_knee" angle { x: -40.0 } }
            angles { name: "right_knee" angle { x: -65.0 } }
        }
    """ % (
        TERRAIN_SIZE,
        terrain_map,
        HULL_X,
        HULL_Z,
        HULL_MASS,
        THIGH_X,
        THIGH_Z,
        LIMB_MASS,
        THIGH_X,
        THIGH_Z,
        LIMB_MASS,
        LEG_X,
        LEG_Z,
        LIMB_MASS,
        LEG_X,
        LEG_Z,
        LIMB_MASS,
        HIP_OFFSET_X,
        HULL_Z,
        -HIP_OFFSET_X,
        THIGH_Z,
        HIP_OFFSET_X,
        HULL_Z,
        -HIP_OFFSET_X,
        THIGH_Z,
        THIGH_Z,
        LEG_Z,
        THIGH_Z,
        LEG_Z,
        STRENGTH_HIP,
        STRENGTH_KNEE,
        STRENGTH_HIP,
        STRENGTH_KNEE,
        INITIAL_X,
        TERRAIN_SIZE / 2,)

if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalker()
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
            print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
            print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
            print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED:
                supporting_knee_angle += 0.03
            supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)

        if terminated or truncated:
            break

