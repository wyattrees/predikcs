from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([Node(
        package='predikcs',
        executable="predikcs_fetch",
        name="KCS_Controller",
        output='screen',
        emulate_tty=True,
        parameters=[{"/KCS_Controller/planning_root_link": "torso_lift_link"}],
        remappings=[("/planning_robot_urdf", "/robot_description"),
                    ("/joint_commands", "/arm_controller/joint_velocity/joint_commands")]

    )])