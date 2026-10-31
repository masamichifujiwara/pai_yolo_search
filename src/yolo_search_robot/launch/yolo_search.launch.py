"""Launch Gazebo + ros_gz_bridge + YOLO detector + search controller (+ recorder / teleop).

  ros2 launch yolo_search_robot yolo_search.launch.py                 # with Gazebo GUI
  ros2 launch yolo_search_robot yolo_search.launch.py headless:=true record:=true \
      video:=/tmp/demo.mp4                                             # no GUI, record a video
  ros2 launch yolo_search_robot yolo_search.launch.py teleop:=true    # + keyboard teleop (xterm)

Structure follows the baseline sample (resource path env var -> gazebo launch include ->
parameter_bridge with a yaml config -> nodes). It works with Gazebo Harmonic (ROS 2 Jazzy).
For Gazebo Fortress (ROS 2 Humble) the SDF plugin names and bridge message types are
rewritten to their "ignition" spelling into a temporary directory at launch time.
"""
import os
import shutil
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'yolo_search_robot'


def gz_flavour():
    v = (os.environ.get('GZ_VERSION') or os.environ.get('IGNITION_VERSION') or '').lower()
    if v:
        return 'fortress' if v in ('fortress', 'edifice', 'citadel') else 'gz'
    return 'fortress' if os.environ.get('ROS_DISTRO') == 'humble' else 'gz'


def to_fortress(text, kind):
    if kind == 'sdf':
        return (text.replace('gz-sim-', 'ignition-gazebo-')
                    .replace('gz::sim::', 'ignition::gazebo::')
                    .replace('gz_frame_id', 'ignition_frame_id'))
    return text.replace('gz.msgs.', 'ignition.msgs.')


def prepare_files(share, flavour):
    """Return (world_path, bridge_yaml, resource_dir) for the detected Gazebo flavour."""
    world = os.path.join(share, 'worlds', 'search_arena.sdf')
    bridge = os.path.join(share, 'config', 'bridge.yaml')
    models = os.path.join(share, 'models')
    if flavour != 'fortress':
        return world, bridge, models
    tmp = tempfile.mkdtemp(prefix='yolo_search_fortress_')
    shutil.copytree(models, os.path.join(tmp, 'models'))
    for root, _d, files in os.walk(os.path.join(tmp, 'models')):
        for f in files:
            if f.endswith('.sdf'):
                p = os.path.join(root, f)
                with open(p) as fh:
                    txt = fh.read()
                with open(p, 'w') as fh:
                    fh.write(to_fortress(txt, 'sdf'))
    out = {}
    for name, src, kind in (('world', world, 'sdf'), ('bridge', bridge, 'yaml')):
        dst = os.path.join(tmp, os.path.basename(src))
        with open(src) as fh:
            txt = fh.read()
        with open(dst, 'w') as fh:
            fh.write(to_fortress(txt, kind))
        out[name] = dst
    return out['world'], out['bridge'], os.path.join(tmp, 'models')


def launch_setup(context):
    share = get_package_share_directory(PKG)
    flavour = gz_flavour()
    world, bridge_yaml, models = prepare_files(share, flavour)
    params = os.path.join(share, 'config', 'params.yaml')

    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'
    record = LaunchConfiguration('record').perform(context).lower() == 'true'
    teleop = LaunchConfiguration('teleop').perform(context).lower() == 'true'
    video = LaunchConfiguration('video').perform(context)
    model = LaunchConfiguration('model').perform(context)
    sim = {'use_sim_time': True}

    resource_path = os.pathsep.join([models, os.path.join(share, 'worlds'),
                                     os.environ.get('GZ_SIM_RESOURCE_PATH', '')])
    actions = [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', resource_path),
    ]

    gz_args = f'-r -v 2 {world}'
    if headless:
        gz_args = '-s --headless-rendering ' + gz_args
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items()))

    actions.append(Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge',
        parameters=[{'config_file': bridge_yaml}, sim], output='screen'))

    actions.append(Node(package=PKG, executable='yolo_detector', name='yolo_detector',
                        parameters=[params, sim, {'model': model}], output='screen'))
    actions.append(Node(package=PKG, executable='search_controller', name='search_controller',
                        parameters=[params, sim], output='screen'))
    if record:
        actions.append(Node(package=PKG, executable='demo_recorder', name='demo_recorder',
                            parameters=[params, sim, {'output': video}], output='screen'))
    if teleop:
        actions.append(Node(package=PKG, executable='keyboard_teleop', name='keyboard_teleop',
                            output='screen', prefix='xterm -e'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run Gazebo server only (no GUI)'),
        DeclareLaunchArgument('record', default_value='false',
                              description='Record the composed demo video'),
        DeclareLaunchArgument('video', default_value=os.path.expanduser('~/yolo_search_demo.mp4'),
                              description='Output path of the demo video'),
        DeclareLaunchArgument('model', default_value='yolo11n.pt',
                              description='YOLO weights (name or path)'),
        DeclareLaunchArgument('teleop', default_value='false',
                              description='Open the keyboard teleop in an xterm'),
        OpaqueFunction(function=launch_setup),
    ])
