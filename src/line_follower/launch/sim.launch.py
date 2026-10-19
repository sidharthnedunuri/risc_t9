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
    robot_file = os.path.join(pkg_line_follower, 'worlds', 'sim_robot.urdf') # Change to .sdf if your robot uses an SDF file

    # Set resource path so Gazebo finds your worlds and models
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_line_follower, '..')
    )

    # Launch Gazebo Sim in headless server mode with headless-rendering enabled for the camera sensor
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r -s --headless-rendering {world_file}'}.items(),
    )

    # Bridge ROS 2 and Gazebo topics (/clock and camera image)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image'
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

    # Start your combined line follower logic node
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
