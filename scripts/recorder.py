#!/usr/bin/env python

import rospy
import json
from rospy.core import is_shutdown
import tf2_ros
import math
import rospkg
import sys
import os
import pyquaternion
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_msgs.msg import Time
from franka_core_msgs.msg import JointCommand


ROBOT_TYPE = 0 #0 for Fetch, 1 for Panda
baseline = False
base_folder_name = "fetch_null_testing"

rospack = rospkg.RosPack()
if(ROBOT_TYPE == 0):
    rewards_params = [-50, -2, 500, -2]
    if(baseline):
        tree_params = [(1, 1, 1), 0.2, 1.0]
    else:
        tree_params = [(2,80,2), 0.2, 0.95]
    base_test_file = rospack.get_path("predikct") + "/test_scripts/fetch_"
else:
    rewards_params = [-20, -1.5, 50, -4]
    if(baseline):
        tree_params = [(1, 1, 1), 0.2, 1.0]
    else:
        tree_params = [(2, 3, 4), 1.0, 0.5]
    base_test_file = rospack.get_path("predikct") + "/test_scripts/panda_"
test_trajectories =  ["tuning_cube"]

loop_rate = 5
ideal_path = []
actual_path = []
waypoint_errors = []
linear_errors = []
angular_errors = []
velocity_commands = []
joint_commands = []
joint_positions = []
controller_info_string = ""
verbose = False
latest_command_tuple = None

class TestRunner:
    def __init__(self, json_object, tf2_buffer):
        self.tf2_buffer = tf2_buffer

        self.waypoint_list = []
        self.waypoint_timings = []
        self.reference_link = json_object["reference_link_name"]
        self.ee_link = json_object["target_link_name"]
        self.first_joint = json_object["first_joint_name"]
        self.last_joint = json_object["last_joint_name"]

        self.velocity_publisher = rospy.Publisher(json_object["velocity_command_topic"], Twist, queue_size=1)
        self.waypoint_publisher = rospy.Publisher("waypoint_updates", PoseStamped, queue_size=1)
        if(ROBOT_TYPE == 0):
            self.joint_command_subscriber = rospy.Subscriber(json_object["joint_command_topic"], JointState, self.receive_joint_command)
        else:
            self.joint_command_subscriber = rospy.Subscriber(json_object["joint_command_topic"], JointCommand, self.receive_joint_command)
        self.joint_state_subscriber = rospy.Subscriber(json_object["joint_state_topic"], JointState, self.receive_joint_state)
        self.request_info_publisher = rospy.Publisher("tester_requests", String, queue_size=1)
        self.get_info_subscriber = rospy.Subscriber("send_controller_info", String, self.receive_controller_info)

        self.start_time = None
        self.next_waypoint_index = 0
        self.trajectory_active = False
        self.motionless = True

        self.control_rate = 1.0 / float(loop_rate)
        self.ideal_pos = [0, 0, 0]
        self.ideal_rot = [0, 0, 0, 1]
        self.last_velocity_command = None
        self.joint_names_start = None
        self.joint_names_end = None
        self.latest_joint_positions = None
        rospy.sleep(1)

    def __update_pose(self):
        last_transform = self.tf2_buffer.lookup_transform(self.reference_link, self.ee_link, rospy.Time(0), rospy.Duration(1))
        self.last_ee_pos = last_transform.transform.translation
        self.last_ee_rot = [last_transform.transform.rotation.x, last_transform.transform.rotation.y, last_transform.transform.rotation.z, last_transform.transform.rotation.w]
    
    def __quat_to_euler(self, quat):
        roll = math.atan2(2*(quat[3]*quat[0] + quat[1]*quat[2]), 1 - 2*(quat[0]**2 + quat[1]**2))
        sinp = 2*(quat[3]*quat[1] - quat[2]*quat[0])
        if(abs(sinp) >= 1):
            if(sinp > 1):
                pitch = math.pi / 2
            else:
                pitch = -math.pi / 2
        else:
            pitch = math.asin(sinp)
        yaw = math.atan2(2*(quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2*(quat[1]**2 + quat[2]**2))

        if(roll < 0): roll = 2*math.pi + roll
        if(pitch < 0): pitch = 2*math.pi + pitch
        if(yaw < 0): yaw = 2*math.pi + yaw

        euler_rot = Vector3()
        euler_rot.x = roll
        euler_rot.y = pitch
        euler_rot.z = yaw
        return euler_rot
    
    def __angle_diff(self, angle1, angle2):
        adiff = angle2 - angle1
        if(abs(adiff) <= math.pi):
            return adiff
        elif(angle2 > angle1):
            return -((2*math.pi - angle2) + angle1)
        else:
            return (2*math.pi - angle1) + angle2
    
    def __quat_diff(self, quat1, quat2):
        return self.__quat_mult(quat2, self.__inv_quat(quat1))
    
    def __quat_mult(self, quat1, quat2):
        w = quat1[3] * quat2[3] - (quat1[0] * quat2[0] + quat1[1] * quat2[1] + quat1[2] * quat2[2])
        x = quat1[3] * quat2[0] + quat1[0] * quat2[3] + quat1[1] * quat2[2] - quat1[2] * quat2[1]
        y = quat1[3] * quat2[1] - quat1[0] * quat2[2] + quat1[1] * quat2[3] + quat1[2] * quat2[0]
        z = quat1[3] * quat2[2] + quat1[0] * quat2[1] - quat1[1] * quat2[0] + quat1[2] * quat2[3]
        return [x, y, z, w]
    
    def __vec_dist(self, vec1, vec2):
        sum_diff = 0
        for i in range(0, len(vec1)):
            sum_diff += (vec1[i] - vec2[i])**2
        return math.sqrt(sum_diff)
    
    def __quat_dist(self, quat1, quat2):
        inner_product = abs(quat1[0]*quat2[0] + quat1[1]*quat2[1] + quat1[2]*quat2[2] + quat1[3]*quat2[3])
        if(inner_product > 1):
            inner_product = 1
        return 2*math.acos(inner_product)

    def __inv_quat(self, quat):
        return [-quat[0], -quat[1], -quat[2], quat[3]]

    def setup_waypoints(self, json_object):
        self.waypoint_list = json_object["end_effector_waypoints"]
        for waypoint in self.waypoint_list:
            quat_mag = math.sqrt(waypoint[3]**2 + waypoint[4]**2 + waypoint[5]**2 + waypoint[6]**2)
            waypoint[3] = waypoint[3] / quat_mag
            waypoint[4] = waypoint[4] / quat_mag
            waypoint[5] = waypoint[5] / quat_mag
            waypoint[6] = waypoint[6] / quat_mag
        self.waypoint_timings = json_object["end_effector_waypoint_timings"]

    def receive_joint_command(self, joint_command_msg):
        global joint_commands
        global latest_command_tuple
        if(not self.trajectory_active):
            return
        latest_command_tuple = (velocity_commands[-1], joint_command_msg.velocity)
    
    def receive_joint_state(self, joint_state_msg):
        if(not self.trajectory_active):
            return
        if(self.joint_names_start is None or self.joint_names_end is None):
            self.joint_names_start = joint_state_msg.name.index(self.first_joint)
            self.joint_names_end = joint_state_msg.name.index(self.last_joint)
        
        self.latest_joint_positions = joint_state_msg.position[self.joint_names_start:self.joint_names_end + 1]

    def receive_controller_info(self, controller_msg):
        global controller_info_string
        controller_info_string = controller_msg.data + ", control rate: " + str(self.control_rate)

    def start_trajectory(self):
        self.start_time = rospy.Time.now().to_sec()
        self.next_waypoint_index = 0
        self.send_new_waypoint(0, 0.0)
        self.send_new_waypoint(1, self.waypoint_timings[0])
        self.trajectory_active = True
        self.__update_pose()
        self.ideal_pos[0] = self.last_ee_pos.x
        self.ideal_pos[1] = self.last_ee_pos.y
        self.ideal_pos[2] = self.last_ee_pos.z
        self.ideal_rot[0] = self.last_ee_rot[0]
        self.ideal_rot[1] = self.last_ee_rot[1]
        self.ideal_rot[2] = self.last_ee_rot[2]
        self.ideal_rot[3] = self.last_ee_rot[3]

    def send_new_waypoint(self, waypoint_to_send, time_til_start):
        waypoint_pose = PoseStamped()
        #TODO: Need to be sending NEXT waypoint with time until that waypoint becomes active
        waypoint_pose.pose.position.x = self.waypoint_list[waypoint_to_send][0]
        waypoint_pose.pose.position.y = self.waypoint_list[waypoint_to_send][1]
        waypoint_pose.pose.position.z = self.waypoint_list[waypoint_to_send][2]
        waypoint_pose.pose.orientation.x = self.waypoint_list[waypoint_to_send][3]
        waypoint_pose.pose.orientation.y = self.waypoint_list[waypoint_to_send][4]
        waypoint_pose.pose.orientation.z = self.waypoint_list[waypoint_to_send][5]
        waypoint_pose.pose.orientation.w = self.waypoint_list[waypoint_to_send][6]
        waypoint_pose.header.stamp = rospy.Time.from_sec(time_til_start)
        self.waypoint_publisher.publish(waypoint_pose)
    
    def end_trajectory(self):
        self.trajectory_active = False

    def process_trajectory(self, command_time):
        global waypoint_errors
        global linear_errors
        global angular_errors
        global ideal_path
        global actual_path
        global velocity_commands
        global joint_positions
        if(not self.trajectory_active):
            return
        self.__update_pose()
        while(self.next_waypoint_index < len(self.waypoint_timings) and (command_time + 0.001) >= self.waypoint_timings[self.next_waypoint_index]):
            last_pos = [self.last_ee_pos.x, self.last_ee_pos.y, self.last_ee_pos.z]
            waypoint_pos = self.waypoint_list[self.next_waypoint_index][0:3]
            last_rot = self.last_ee_rot
            waypoint_rot = self.waypoint_list[self.next_waypoint_index][3:]
            linear_waypoint_error = self.__vec_dist(last_pos, waypoint_pos)
            angular_waypoint_error = self.__quat_dist(last_rot, waypoint_rot)
            self.next_waypoint_index += 1
            if(self.next_waypoint_index < len(self.waypoint_timings) - 1):
                self.send_new_waypoint(self.next_waypoint_index + 1, self.waypoint_timings[self.next_waypoint_index] - self.waypoint_timings[self.next_waypoint_index - 1])
            waypoint_errors.append((linear_waypoint_error, angular_waypoint_error, last_pos, waypoint_pos, last_rot, waypoint_rot))
        if(self.next_waypoint_index >= len(self.waypoint_timings)):
            self.end_trajectory()
            return
        
        true_time_remaining = (self.waypoint_timings[self.next_waypoint_index] - command_time)
        effective_time_remaining = true_time_remaining - self.control_rate
        if(effective_time_remaining < 0.001):
            effective_time_remaining = 0.0
        at_waypoint = False
        if(effective_time_remaining <= 0):
            at_waypoint = True
            effective_time_remaining = 0
            self.motionless = True
        
        current_waypoint = self.waypoint_list[self.next_waypoint_index]
        next_ideal_pos = [i for i in self.ideal_pos]
        next_ideal_rot = [j for j in self.ideal_rot]

        vel_command = Twist()
        if(at_waypoint):
            vel_command.linear.x = 0
            vel_command.linear.y = 0
            vel_command.linear.z = 0
            vel_command.angular.x = 0
            vel_command.angular.y = 0
            vel_command.angular.z = 0
            next_ideal_pos[0] = current_waypoint[0]
            next_ideal_pos[1] = current_waypoint[1]
            next_ideal_pos[2] = current_waypoint[2]
            next_ideal_rot[0] = current_waypoint[3]
            next_ideal_rot[1] = current_waypoint[4]
            next_ideal_rot[2] = current_waypoint[5]
            next_ideal_rot[3] = current_waypoint[6]
        else:
            vel_command.linear.x = (current_waypoint[0] - self.last_ee_pos.x) / effective_time_remaining
            vel_command.linear.y = (current_waypoint[1] - self.last_ee_pos.y) / effective_time_remaining
            vel_command.linear.z = (current_waypoint[2] - self.last_ee_pos.z) / effective_time_remaining
            if(not self.motionless):
                next_ideal_pos[0] = self.ideal_pos[0] + (current_waypoint[0] - self.ideal_pos[0]) * (self.control_rate / true_time_remaining)
                next_ideal_pos[1] = self.ideal_pos[1] + (current_waypoint[1] - self.ideal_pos[1]) * (self.control_rate / true_time_remaining)
                next_ideal_pos[2] = self.ideal_pos[2] + (current_waypoint[2] - self.ideal_pos[2]) * (self.control_rate / true_time_remaining)
                q_current = pyquaternion.Quaternion(w=self.ideal_rot[3], x=self.ideal_rot[0], y=self.ideal_rot[1], z=self.ideal_rot[2])
                q_waypoint = pyquaternion.Quaternion(w=current_waypoint[6], x=current_waypoint[3], y=current_waypoint[4], z=current_waypoint[5])
                q_slerp = pyquaternion.Quaternion.slerp(q_current, q_waypoint, self.control_rate / true_time_remaining)
                next_ideal_rot[0] = q_slerp.elements[1]
                next_ideal_rot[1] = q_slerp.elements[2]
                next_ideal_rot[2] = q_slerp.elements[3]
                next_ideal_rot[3] = q_slerp.elements[0]

            quat_rotation_needed = self.__quat_diff(self.last_ee_rot, current_waypoint[3:])
            angle_quat = pyquaternion.Quaternion(w=quat_rotation_needed[3], x=quat_rotation_needed[0], y=quat_rotation_needed[1], z=quat_rotation_needed[2])
            angle_remaining = angle_quat.radians * angle_quat.axis
            vel_command.angular.x = angle_remaining[0] / effective_time_remaining
            vel_command.angular.y = angle_remaining[1] / effective_time_remaining
            vel_command.angular.z = angle_remaining[2] / effective_time_remaining
            self.motionless = False

        self.velocity_publisher.publish(vel_command)

        # Record trajectories
        ideal_rotation_distance = self.__quat_dist(self.ideal_rot, current_waypoint[3:])
        actual_rotation_distance = self.__quat_dist(self.last_ee_rot, current_waypoint[3:])
        linear_error = self.__vec_dist([self.last_ee_pos.x, self.last_ee_pos.y, self.last_ee_pos.z], self.ideal_pos)
        angular_error = abs(ideal_rotation_distance - actual_rotation_distance)

        if(verbose):
            print("Current pose: " + str(self.last_ee_pos.x) + ", " + str(self.last_ee_pos.y) + ", " + str(self.last_ee_pos.z) + "/ " + str(self.last_ee_rot))
            print("Current joint positions: " + str(self.latest_joint_positions))
            print("Current waypoint: " + str(current_waypoint))
            print("Ideal rot: " + str(self.ideal_rot) + ", Ideal pos: " + str(self.ideal_pos))
            print("Ideal rot dist: " + str(ideal_rotation_distance) + ", actual rot dist: " + str(actual_rotation_distance) + "rot dist error: " + str(angular_error) + "// Ideal pos: " + str(self.ideal_pos) + " pos error: " + str(linear_error))
            print("Effective Time remaining: " + str(effective_time_remaining))
            print("Velocity command: " + str(vel_command.linear.x) + ", " + str(vel_command.linear.y) + ", " +str(vel_command.linear.z) + ", " + str(vel_command.angular.x) + ", " + str(vel_command.angular.y) + ", " + str(vel_command.angular.z) )


        ideal_lin_error = self.__vec_dist(self.ideal_pos, current_waypoint[0:3])
        actual_lin_error = self.__vec_dist([self.last_ee_pos.x, self.last_ee_pos.y, self.last_ee_pos.z], current_waypoint[0:3])
        ideal_path.append((self.ideal_pos, self.ideal_rot, ideal_lin_error, ideal_rotation_distance))
        actual_path.append(([self.last_ee_pos.x, self.last_ee_pos.y, self.last_ee_pos.z], self.last_ee_rot, actual_lin_error, actual_rotation_distance))
        linear_errors.append(linear_error)
        angular_errors.append(angular_error)
        velocity_commands.append(([vel_command.linear.x, vel_command.linear.y, vel_command.linear.z], [vel_command.angular.x, vel_command.angular.y, vel_command.angular.z]))
        joint_positions.append(self.latest_joint_positions)
        joint_commands.append(latest_command_tuple)

        # Update ideal position and rotation
        self.ideal_pos = next_ideal_pos
        self.ideal_rot = next_ideal_rot

def main():
    rospy.init_node("test_node")

    #Need to change this manually
    if(ROBOT_TYPE == 0):
        resetter = FetchResetter()
    else:
        resetter = PandaResetter()
    rospy.sleep(1)
    resetter.reset()
    rospy.sleep(5)

    json_test_spec = None
    with open(base_test_file + "parameters.json") as test_spec:
        json_test_spec = json.loads(test_spec.read())
    tf2_buffer = tf2_ros.Buffer()
    tf2_listener = tf2_ros.TransformListener(tf2_buffer)
    num_test_iterations = int(rospy.get_param('~num_tests', 1))
    print("number of iterations: " + str(num_test_iterations))

    test_runner = TestRunner(json_test_spec, tf2_buffer)
    rate = rospy.Rate(loop_rate)

    # Publish initial reward spec
    reward_command_spec = "update_rewards/"
    for param in rewards_params:
        if(hasattr(param, '__iter__')):
            for p in param:
                reward_command_spec += str(p) + "/"
        else:
            reward_command_spec += str(param) + "/"
    test_runner.request_info_publisher.publish(String(reward_command_spec))
    rospy.sleep(2)
    # Publish initial tree spec
    tree_command_spec = "update_tree/"
    for param in tree_params:
        if(hasattr(param, '__iter__')):
            for p in param:
                tree_command_spec += str(p) + "/"
        else:
            tree_command_spec += str(param) + "/"
    test_runner.request_info_publisher.publish(String(tree_command_spec))
    rospy.sleep(2)

    overall_linears = []
    overall_angulars = []
    overall_linear_waypoints = []
    overall_angular_waypoints = []
    
    if not os.path.exists(base_folder_name):
        os.makedirs(base_folder_name)
    test_runner.request_info_publisher.publish(String("request_info/"))
    rospy.sleep(1)
    for test_type in test_trajectories:
        json_waypoints = None
        with open(base_test_file + test_type + ".json") as waypoint_spec:
            json_waypoints = json.loads(waypoint_spec.read())
        test_runner.setup_waypoints(json_waypoints)
        for i in range(num_test_iterations):
            reset_data_recorders()
            test_runner.request_info_publisher.publish(String("new_task/"))
            rospy.sleep(1)
            test_runner.start_trajectory()
            current_time = 0.0
            while(test_runner.trajectory_active and not rospy.is_shutdown()):
                test_runner.process_trajectory(current_time)
                rate.sleep()
                current_time += 1.0 / loop_rate
            
            rospy.sleep(1)

            overall_linears.append(sum(linear_errors))
            overall_angulars.append(sum(angular_errors))
            overall_linear_waypoints.append(sum([pair[0] for pair in waypoint_errors]))
            overall_angular_waypoints.append(sum([pair[1] for pair in waypoint_errors]))
            print("Results summary:")
            print("Linear Error Sum: " + str(overall_linears[-1]))
            print("Angular Error Sum: " + str(overall_angulars[-1]))
            print("Linear Waypoint Errors: " + str(overall_linear_waypoints[-1]))
            print("Angular Waypoint Errors: " + str(overall_angular_waypoints[-1]))
            result_object = {}
            result_object["controller_spec"] = controller_info_string
            result_object["ideal_path"] = ideal_path
            result_object["actual_path"] = actual_path
            result_object["linear_errors"] = linear_errors
            result_object["angular_errors"] = angular_errors
            result_object["velocity_commands"] = velocity_commands
            result_object["joint_commands"] = joint_commands
            result_object["joint_positions"] = joint_positions
            result_object["waypoint_errors"] = waypoint_errors
            with open(base_folder_name + "/" + test_type + "_" + str(i) + ".txt", "w") as result_file:
                result_file.write(json.dumps(result_object))
                result_file.close()
            
            resetter.reset()
            rospy.sleep(5)

def reset_data_recorders():
    global ideal_path
    global actual_path
    global linear_errors
    global angular_errors
    global velocity_commands
    global joint_commands
    global joint_positions
    global waypoint_errors
    ideal_path = []
    actual_path = []
    linear_errors = []
    angular_errors = []
    velocity_commands = []
    joint_commands = []
    joint_positions = []
    waypoint_errors = []

class FetchResetter():
    def __init__(self):
        # Lists of joint angles in the same order as in joint_names
        self.reset_pose = [0.0, -0.5, 0.0, 0.7, 0.0, -0.2, 0.0]

        self.reset_publisher = rospy.Publisher("/arm_controller/joint_velocity/reset_positions", JointState, queue_size=1)

    def reset(self):
        reset_command = JointState()
        reset_command.position = self.reset_pose

        self.reset_publisher.publish(reset_command)

class PandaResetter():
    def __init__(self):
        self.reset_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
        self.reset_publisher = rospy.Publisher("/panda_joint_commands", JointCommand, queue_size=1)
    
    def reset(self):
        print("Resetting panda...")
        reset_command = JointCommand()
        reset_command.mode = 1
        reset_command.names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        reset_command.position = self.reset_pose

        for i in range(10):
            self.reset_publisher.publish(reset_command)
            rospy.sleep(0.1)


if __name__ == '__main__':
    main()
