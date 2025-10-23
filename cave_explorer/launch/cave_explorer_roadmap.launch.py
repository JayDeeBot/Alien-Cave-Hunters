from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare the 'use_sim_time' and 'use_path_planner' arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_path_planner = LaunchConfiguration('use_path_planner', default='false')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # declare_use_path_planner = DeclareLaunchArgument(
    #     'use_path_planner',
    #     default_value='false',
    #     description='Launch the PRM-integrated path planner node if true'
    # )

    # Launch the roadmap builder node
    roadmap_builder_node = Node(
        package='cave_explorer',
        executable='roadmap_builder',
        name='roadmap_builder',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # # Launch the roadmap path node with remapping
    # roadmap_path_node = Node(
    #     package='cave_explorer',
    #     executable='roadmap_path',
    #     name='roadmap_path_node',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}],
    #     condition=IfCondition(use_path_planner),
    #     remappings=[
    #         ('/odom', '/odometry'),   # remap to actual odometry
    #         ('/cmd_vel_prm', '/cmd_vel')
    #     ],
    # )

    return LaunchDescription([
        declare_use_sim_time,
        # declare_use_path_planner,
        roadmap_builder_node,
        # roadmap_path_node
    ])

