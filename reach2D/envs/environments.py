
import pybullet

import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
import pybullet_data
from pybullet_utils import bullet_client
from pkg_resources import parse_version

class XmlBasedRobot:
	"""
	Base class for mujoco .xml based agents.
	"""

	self_collision = True
	def __init__(self,  robot_name, action_dim, obs_dim, self_collision):
		self.parts = None
		self.objects = []
		self.jdict = None
		self.ordered_joints = None
		self.robot_body = None

		high = np.ones([action_dim])
		self.action_space = gym.spaces.Box(-high, high)
		high = np.inf * np.ones([obs_dim])
		self.observation_space = gym.spaces.Box(-high, high)

		#self.model_xml = model_xml
		self.robot_name = robot_name
		self.self_collision = self_collision

	def addToScene(self, bullet_client, bodies):
		self._p = bullet_client

		if self.parts is not None:
			parts = self.parts
		else:
			parts = {}

		if self.jdict is not None:
			joints = self.jdict
		else:
			joints = {}

		if self.ordered_joints is not None:
			ordered_joints = self.ordered_joints
		else:
			ordered_joints = []

		if np.isscalar(bodies):	# streamline the case where bodies is actually just one body
			bodies = [bodies]

		dump = 0
		for i in range(len(bodies)):
			if self._p.getNumJoints(bodies[i]) == 0:
				part_name, robot_name = self._p.getBodyInfo(bodies[i])
				self.robot_name = robot_name.decode("utf8")
				part_name = part_name.decode("utf8")
				parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
			for j in range(self._p.getNumJoints(bodies[i])):
				self._p.setJointMotorControl2(bodies[i],j,pybullet.POSITION_CONTROL,positionGain=0.1,velocityGain=0.1,force=0)
				jointInfo = self._p.getJointInfo(bodies[i], j)
				joint_name=jointInfo[1]
				part_name=jointInfo[12]

				joint_name = joint_name.decode("utf8")
				part_name = part_name.decode("utf8")

				if dump: print("ROBOT PART '%s'" % part_name)
				if dump: print("ROBOT JOINT '%s'" % joint_name)  # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )

				parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

				if part_name == self.robot_name:
					self.robot_body = parts[part_name]

				if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
					parts[self.robot_name] = BodyPart(self._p, self.robot_name, bodies, 0, -1)
					self.robot_body = parts[self.robot_name]

				if joint_name[:6] == "ignore":
					Joint(self._p, joint_name, bodies, i, j).disable_motor()
					continue

				if joint_name[:8] != "jointfix":
					joints[joint_name] = Joint(self._p, joint_name, bodies, i, j)
					ordered_joints.append(joints[joint_name])

					joints[joint_name].power_coef = 100.0

				# TODO: Maybe we need this
				# joints[joint_name].power_coef, joints[joint_name].max_velocity = joints[joint_name].limits()[2:4]
				# self.ordered_joints.append(joints[joint_name])
				# self.jdict[joint_name] = joints[joint_name]

		return parts, joints, ordered_joints, self.robot_body

	def reset_pose(self, position, orientation):
		self.parts[self.robot_name].reset_pose(position, orientation)

class MJCFBasedRobot(XmlBasedRobot):
	"""
	Base class for mujoco .xml based agents.
	"""

	def __init__(self,  model_xml, robot_name, action_dim, obs_dim, self_collision=True):
		XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
		self.model_xml = model_xml
		self.doneLoading=0
	def reset(self, bullet_client):
		
		self._p = bullet_client	
		#print("Created bullet_client with id=", self._p._client)
		if (self.doneLoading==0):
			self.ordered_joints = []
			self.doneLoading=1
			if self.self_collision:
				self.objects = self._p.loadMJCF(os.path.join(pybullet_data.getDataPath(),"mjcf", self.model_xml), flags=pybullet.URDF_USE_SELF_COLLISION|pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
				self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects	)
			else:
				self.objects = self._p.loadMJCF(os.path.join(pybullet_data.getDataPath(),"mjcf", self.model_xml))
				self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
		self.robot_specific_reset(self._p)

		s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

		return s

	def calc_potential(self):
		return 0


class Pose_Helper: # dummy class to comply to original interface
	def __init__(self, body_part):
		self.body_part = body_part

	def xyz(self):
		return self.body_part.current_position()

	def rpy(self):
		return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

	def orientation(self):
		return self.body_part.current_orientation()

class BodyPart:
	def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
		self.bodies = bodies
		self._p = bullet_client
		self.bodyIndex = bodyIndex
		self.bodyPartIndex = bodyPartIndex
		self.initialPosition = self.current_position()
		self.initialOrientation = self.current_orientation()
		self.bp_pose = Pose_Helper(self)

	def state_fields_of_pose_of(self, body_id, link_id=-1):  # a method you will most probably need a lot to get pose and orientation
		if link_id == -1:
			(x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
		else:
			(x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
		return np.array([x, y, z, a, b, c, d])

	def get_pose(self):
		return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

	def speed(self):
		if self.bodyPartIndex == -1:
			(vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
		else:
			(x,y,z), (a,b,c,d), _,_,_,_, (vx, vy, vz), (vr,vp,vy) = self._p.getLinkState(self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
		return np.array([vx, vy, vz])

	def current_position(self):
		return self.get_pose()[:3]

	def current_orientation(self):
		return self.get_pose()[3:]

	def get_orientation(self):
		return self.current_orientation()

	def reset_position(self, position):
		self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, self.get_orientation())

	def reset_orientation(self, orientation):
		self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(), orientation)

	def reset_velocity(self, linearVelocity=[0,0,0], angularVelocity =[0,0,0]):
		self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

	def reset_pose(self, position, orientation):
		self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

	def pose(self):
		return self.bp_pose

	def contact_list(self):
		return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:
	def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
		self.bodies = bodies
		self._p = bullet_client
		self.bodyIndex = bodyIndex
		self.jointIndex = jointIndex
		self.joint_name = joint_name
		
		jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
		self.lowerLimit = jointInfo[8]
		self.upperLimit = jointInfo[9]
		
		self.power_coeff = 0

	def set_state(self, x, vx):
		self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

	def current_position(self): # just some synonyme method
		return self.get_state()

	def current_relative_position(self):
		pos, vel = self.get_state()
		pos_mid = 0.5 * (self.lowerLimit + self.upperLimit);
		return (
			2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit),
			0.1 * vel
		)

	def get_state(self):
		x, vx,_,_ = self._p.getJointState(self.bodies[self.bodyIndex],self.jointIndex)
		return x, vx

	def get_position(self):
		x, _ = self.get_state()
		return x

	def get_orientation(self):
		_,r = self.get_state()
		return r

	def get_velocity(self):
		_, vx = self.get_state()
		return vx

	def set_position(self, position):
		self._p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,pybullet.POSITION_CONTROL, targetPosition=position)

	def set_velocity(self, velocity):
		self._p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,pybullet.VELOCITY_CONTROL, targetVelocity=velocity)

	def set_motor_torque(self, torque): # just some synonyme method
		self.set_torque(torque)

	def set_torque(self, torque):
		self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex], jointIndex=self.jointIndex, controlMode=pybullet.TORQUE_CONTROL, force=torque) #, positionGain=0.1, velocityGain=0.1)

	def reset_current_position(self, position, velocity): # just some synonyme method
		self.reset_position(position, velocity)

	def reset_position(self, position, velocity):
		self._p.resetJointState(self.bodies[self.bodyIndex],self.jointIndex,targetValue=position, targetVelocity=velocity)
		self.disable_motor()

	def disable_motor(self):
		self._p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,controlMode=pybullet.POSITION_CONTROL, targetPosition=0, targetVelocity=0, positionGain=0.1, velocityGain=0.1, force=0)



class Reacher(MJCFBasedRobot):
	TARG_LIMIT = 0.27

	def __init__(self):
		MJCFBasedRobot.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=9)

	def robot_specific_reset(self, bullet_client):
		self.jdict["target_x"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["target_y"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.fingertip = self.parts["fingertip"]
		self.target = self.parts["target"]
		self.central_joint = self.jdict["joint0"]
		self.elbow_joint = self.jdict["joint1"]
		self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

	def apply_action(self, a):
		assert (np.isfinite(a).all())
		self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
		self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

	def calc_state(self):
		theta, self.theta_dot = self.central_joint.current_relative_position()
		self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
		target_x, _ = self.jdict["target_x"].current_position()
		target_y, _ = self.jdict["target_y"].current_position()
		self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
		return np.array([
			target_x,
			target_y,
			self.to_target_vec[0],
			self.to_target_vec[1],
			np.cos(theta),
			np.sin(theta),
			self.theta_dot,
			self.gamma,
			self.gamma_dot,
		])

	def calc_potential(self):
		return -100 * np.linalg.norm(self.to_target_vec)



class MJCFBaseBulletEnv(gym.Env):
	"""
	Base class for Bullet physics simulation loading MJCF (MuJoCo .xml) environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
		}

	def __init__(self, robot, render=False):
		self.scene = None
		self.physicsClientId = -1
		self.ownsPhysicsClient = 0
		
		self.isRender = render
		self.robot = robot
		self._seed()
		self._cam_dist = 3
		self._cam_yaw = 0
		self._cam_pitch = -30
		self._render_width =320
		self._render_height = 240

		self.action_space = robot.action_space
		self.observation_space = robot.observation_space
	def configure(self, args):
		self.robot.args = args
	def _seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		self.robot.np_random = self.np_random # use the same np_randomizer for robot as for env
		return [seed]

	def _reset(self):
		if (self.physicsClientId<0):
			self.ownsPhysicsClient = True


			if self.isRender:
      				self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
			else:
				self._p = bullet_client.BulletClient()

			self.camera = Camera(self._p)
	
			self.physicsClientId = self._p._client
			self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

		if self.scene is None:
			self.scene = self.create_single_player_scene(self._p)
		if not self.scene.multiplayer and self.ownsPhysicsClient:
			self.scene.episode_restart(self._p)

		self.robot.scene = self.scene

		self.frame = 0
		self.done = 0
		self.reward = 0
		dump = 0
		s = self.robot.reset(self._p)
		self.potential = self.robot.calc_potential()
		return s

	def _render(self, mode, close=False):
		if (mode=="human"):
			self.isRender = True
		if mode != "rgb_array":
			return np.array([])

		base_pos=[0,0,0]
		if (hasattr(self,'robot')):
			if (hasattr(self.robot,'body_xyz')):
				base_pos = self.robot.body_xyz

		view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=base_pos,
			distance=self._cam_dist,
			yaw=self._cam_yaw,
			pitch=self._cam_pitch,
			roll=0,
			upAxisIndex=2)
		proj_matrix = self._p.computeProjectionMatrixFOV(
			fov=60, aspect=float(self._render_width)/self._render_height,
			nearVal=0.1, farVal=100.0)
		(_, _, px, _, _) = self._p.getCameraImage(
		width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
			projectionMatrix=proj_matrix,
			renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
			)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array


	def _close(self):
		if (self.ownsPhysicsClient):
			if (self.physicsClientId>=0):
				self._p.disconnect()
		self.physicsClientId = -1

	def HUD(self, state, a, done):
		pass

	# backwards compatibility for gym >= v0.9.x
	# for extension of this class.
	def step(self, *args, **kwargs):
		if self.isRender:
			base_pos=[0,0,0]
			if (hasattr(self,'robot')):
				if (hasattr(self.robot,'body_xyz')):
					base_pos = self.robot.body_xyz
					# Keep the previous orientation of the camera set by the user.
					#[yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
					self._p.resetDebugVisualizerCamera(3,0,0, base_pos)


		return self._step(*args, **kwargs)

	if parse_version(gym.__version__)>=parse_version('0.9.6'):
		close = _close
		render = _render
		reset = _reset
		seed = _seed


class Camera:
	def __init__(self,_p):
		self._p = _p

	def move_and_look_at(self,i,j,k,x,y,z):
		lookat = [x,y,z]
		distance = 0.5
		yaw = 0
		self._p.resetDebugVisualizerCamera(distance, yaw, -50, lookat)


class Scene:
    "A base class for single- and multiplayer scenes"

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.np_random, seed = gym.utils.seeding.np_random(None)
        self.timestep = timestep
        self.frame_skip = frame_skip

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(self._p, gravity, timestep, frame_skip)

        self.test_window_still_open = True  # or never opened
        self.human_render_detected = False  # if user wants render("human"), we open test window

        self.multiplayer_robots = {}

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def actor_introduce(self, robot):
        "Usually after scene reset"
        if not self.multiplayer: return
        self.multiplayer_robots[robot.player_n] = robot

    def actor_is_active(self, robot):
        """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
        return not self.multiplayer

    def episode_restart(self, bullet_client):
        "This function gets overridden by specific scene, to reset specific objects into their start positions"
        self.cpp_world.clean_everything()
        #self.cpp_world.test_window_history_reset()

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step(self.frame_skip)

class SingleRobotEmptyScene(Scene):
    multiplayer = False  # this class is used "as is" for InvertedPendulum, Reacher

class World:

	def __init__(self, bullet_client, gravity, timestep, frame_skip):
		self._p = bullet_client
		self.gravity = gravity
		self.timestep = timestep
		self.frame_skip = frame_skip
		self.numSolverIterations = 5
		self.clean_everything()
		
		
	def clean_everything(self):
		#p.resetSimulation()
		self._p.setGravity(0, 0, -self.gravity)
		self._p.setDefaultContactERP(0.9)
		#print("self.numSolverIterations=",self.numSolverIterations)
		self._p.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip, numSolverIterations=self.numSolverIterations, numSubSteps=self.frame_skip)

	def step(self, frame_skip):
		self._p.stepSimulation()



class ReacherBulletEnv(MJCFBaseBulletEnv):
	def __init__(self):
		self.robot = Reacher()
		MJCFBaseBulletEnv.__init__(self, self.robot)

	def create_single_player_scene(self, bullet_client):
		return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

	def _step(self, a):
		assert (not self.scene.multiplayer)
		self.robot.apply_action(a)
		self.scene.global_step()

		state = self.robot.calc_state()  # sets self.to_target_vec

		potential_old = self.potential
		self.potential = self.robot.calc_potential()

		electricity_cost = (
			-0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot))  # work torque*angular_velocity
			- 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
		)
		stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
		self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
		self.HUD(state, a, False)
		return state, sum(self.rewards), False, {}

	def camera_adjust(self):
		x, y, z = self.robot.fingertip.pose().xyz()
		x *= 0.5
		y *= 0.5
		self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)