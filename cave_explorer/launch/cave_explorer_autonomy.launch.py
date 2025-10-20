
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ld = LaunchDescription()
    config_path = [FindPackageShare('cave_explorer'), 'config']
    
    # Additional command line arguments
    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time'
    )
    print_feedback_launch_arg = DeclareLaunchArgument(
        'print_feedback',
        default_value='False',
        description='Flag to enable print feedback from action server'
    )

    # Start Navigation Stack
    cave_explorer_node = Node(
        package='cave_explorer',
        executable='cave_explorer',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                     'print_feedback': LaunchConfiguration('print_feedback'),

                    #  ### Original Computer Vision Model ###
                    # 'computer_vision_model_filename': PathJoinSubstitution(config_path+['stop_data.xml']),
                    
                    ### Perception 2 ###
                    # --- YOLO params ---
                    'yolo_model_path': PathJoinSubstitution(config_path + ['best.pt']),
                    'yolo_conf': 0.05,
                    'yolo_iou': 0.50,
                    'yolo_imgsz': 640,
                    'yolo_classes': ['mushroom','green_crystal','alien','white_sphere','ice_castle','stop_sign'],
                    'yolo_allowed_class_names': ['mushroom','green_crystal'],

                    ### Perception 3 ###
                    # Enable depth-based localisation
                    'use_depth_for_localisation': True,

                    }]
    )

    ld.add_action(use_sim_time_launch_arg)
    ld.add_action(print_feedback_launch_arg)
    ld.add_action(cave_explorer_node)

    return ld