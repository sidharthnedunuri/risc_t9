import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_line_follower = get_package_share_directory('line_follower')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(pkg_line_follower, 'worlds', 'circular_world_track.sdf')
    # Updated to point to your final URDF filename inside the package/worlds or install directory
    robot_file = os.path.join(pkg_line_follower, 'worlds', 'line_follower_robot.urdf')

    # Set resource path so Gazebo finds your worlds and models
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_line_follower, '..')
    )

    # Launch Gazebo Sim with GUI enabled (-r runs it immediately)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # Bridge ROS 2 and Gazebo topics (Camera, Clock, and Velocity Commands)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
        ],
        output='screen'
    )

    # Spawn the robot on the track starting point
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', robot_file,
            '-name', 'line_follower_robot',
            '-x', '0.0', '-y', '0.0', '-z', '0.1'
        ],
        output='screen'
    )

    # Start your line follower logic node
    line_follower_node = Node(
        package='line_follower',
        executable='line_follower',
        output='screen'
    )

    return LaunchDescription([
        set_gz_resource_path,
        gz_sim,
        bridge,
        spawn_robot,
        line_follower_node,
    ])
