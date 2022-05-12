#!/usr/bin/env python

import rospy
import json
from rospy.core import is_shutdown
import tf2_ros
import sys
import os
import pyquaternion
import signal
import copy
import math
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_msgs.msg import Time
from control_msgs.msg import GripperCommandActionGoal

reference_link = "torso_lift_link"
ee_link = "gripper_link"
joint_state_topic = "/joint_states"
joint_command_topic = "/arm_controller/joint_velocity/joint_commands"
user_command_topic = "/teleop_commands"
gripper_command_topic = "/gripper_controller/gripper_action/goal"
record_rate = 20 # in Hz
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
joint_continuous = [False, False, True, False, True, False, True]
base_folder_name = "PrediKCS_Exp_Data/"

data_timesteps = []
last_joint_positions = None
last_joint_velocities = None
last_joint_accelerations = None
active = True

def handler(signum, frame):
	global active
	print("Saving file...")
	active = False

signal.signal(signal.SIGINT, handler)

class DataListener:
	def __init__(self, ref_link, pose_link, tf2_listen, state_topic, vels_command_topic, twist_topic, gripper_topic):
		self.ref_link = ref_link
		self.pose_link = pose_link
		self.tf2_buffer = tf2_listen
		self.last_joint_state = None
		self.last_ee_pose = None
		self.last_joint_vels_command = None
		self.last_twist_command = None
		self.heard_first_command = False
		self.heard_gripper_command = False

		self.joint_state_subscriber = rospy.Subscriber(state_topic, JointState, self.receive_joint_state)
		self.joint_command_subscriber = rospy.Subscriber(vels_command_topic, JointState, self.receive_joint_command)
		self.user_command_subscriber = rospy.Subscriber(twist_topic, Twist, self.receive_user_command)
		self.gripper_command_subscriber = rospy.Subscriber(gripper_topic, GripperCommandActionGoal, self.receive_gripper_command)

	def receive_joint_state(self, msg):
		new_joint_state = []
		j = 0
		for i in range(len(msg.position)):
			if(j < len(joint_names) and msg.name[i] == joint_names[j]):
				new_joint_state.append(round(msg.position[i], 4))
				j += 1
		if(len(new_joint_state) != len(joint_names)):
			pass
		else:
			self.last_joint_state = new_joint_state
			last_transform = self.tf2_buffer.lookup_transform(self.ref_link, self.pose_link, rospy.Time(0), rospy.Duration(1))
			new_ee_pose = [last_transform.transform.translation.x, last_transform.transform.translation.y, last_transform.transform.translation.z,
				last_transform.transform.rotation.x, last_transform.transform.rotation.y, last_transform.transform.rotation.z, last_transform.transform.rotation.w]
			self.last_ee_pose = new_ee_pose

	def receive_joint_command(self, msg):
		self.last_joint_vels_command = msg.velocity

	def receive_user_command(self, msg):
		self.heard_first_command = True
		new_twist_command = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z]
		self.last_twist_command = new_twist_command

	def receive_gripper_command(self, msg):
		self.heard_gripper_command = True
		self.heard_first_command = False

# Example data timestep format:
'''
data_timestep = {
	"joint_positions" : [joint position vector]
	"joint_velocities" : [joint velocities]
	"joint accelerations" : [joint accelerations]
	"joint jerk" : [joint jerk]
	"ee_pose" : [ee vec pos + quat]
	"user_command" : [twist command]
	"joint_command" : [joint velocities command]
	"gripper_command" : True or False
	"zero_command" : True or False
}
'''
def GetDataStep(listener):
	global last_joint_positions
	global last_joint_velocities
	global last_joint_accelerations

	if(not listener.heard_first_command):
		return None

	data_dict = {}
	joint_pos = listener.last_joint_state
	joint_vel = listener.last_joint_vels_command
	user_twist = listener.last_twist_command
	pose = listener.last_ee_pose
	gripped = listener.heard_gripper_command

	# Raw data values
	data_dict["joint_positions"] = copy.deepcopy(listener.last_joint_state)
	data_dict["ee_pose"] = copy.deepcopy(listener.last_ee_pose)
	data_dict["user_command"] = copy.deepcopy(listener.last_twist_command)
	listener.last_twist_command = None
	data_dict["joint_command"] = copy.deepcopy(listener.last_joint_vels_command)
	data_dict["gripper_command"] = listener.heard_gripper_command
	listener.heard_gripper_command = False

	# Calculated values
	all_zeros = True
	if(not data_dict["user_command"] is None):
		for user_vel in data_dict["user_command"]:
			if(abs(user_vel) > 0.0001):
				all_zeros = False
				break
	data_dict["zero_command"] = all_zeros
	data_dict["joint_velocities"] = [0] * len(joint_names)
	data_dict["joint_accelerations"] = [0] * len(joint_names)
	data_dict["joint_jerks"] = [0] * len(joint_names)
	if(not(last_joint_positions is None)):
		for i in range(len(joint_names)):
			joint_dist = round(data_dict["joint_positions"][i] - last_joint_positions[i], 4)
			if(abs(joint_dist) > math.pi and joint_continuous[i]):
				joint_dist = round(math.copysign(abs(joint_dist - (2 * math.pi)), -1 * joint_dist), 4)
			timestep = 1.0 / record_rate
			new_vel = round(joint_dist / timestep, 4)
			new_accel = round((new_vel - last_joint_velocities[i]) / timestep, 4)
			new_jerk = round((new_accel - last_joint_accelerations[i]) / timestep, 4)
			data_dict["joint_velocities"][i] = new_vel
			data_dict["joint_accelerations"][i] = new_accel
			data_dict["joint_jerks"][i] = new_jerk

	last_joint_positions = copy.deepcopy(data_dict["joint_positions"])
	last_joint_velocities = copy.deepcopy(data_dict["joint_velocities"])
	last_joint_accelerations = copy.deepcopy(data_dict["joint_accelerations"])
	return data_dict

def main():
	global data_timesteps
	rospy.init_node("experiment_recorder")
	rospy.sleep(1)
	looper = rospy.Rate(record_rate)

	tf2_buffer = tf2_ros.Buffer()
	tf2_listener = tf2_ros.TransformListener(tf2_buffer)
	p_id = rospy.get_param('~pid', "p0000")
	c_type = rospy.get_param('~type', "unknown_controller")
	task_name = rospy.get_param('~task', "unknown_task")
	controller_type = rospy.get_param("/teleop_type")
	dl = DataListener(reference_link, ee_link, tf2_buffer, joint_state_topic, joint_command_topic, user_command_topic, gripper_command_topic)
	rate = rospy.Rate(record_rate)
	while(active and not rospy.is_shutdown()):
		data_step = GetDataStep(dl)
		if(not data_step is None):
			data_timesteps.append(data_step)
		try:
			looper.sleep()
		except:
			break
	folder_name = base_folder_name + p_id
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	data_json = {
		"controller_type" : controller_type,
		"data_steps" : data_timesteps
	}
	with open(folder_name + "/" + p_id + "_" + c_type + "_" + task_name + ".txt", "w") as result_file:
		result_file.write(json.dumps(data_json))
		result_file.close()


if __name__ == '__main__':
	main()