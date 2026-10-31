import os
from glob import glob

from setuptools import setup

package_name = 'yolo_search_robot'


def tree(src_dir):
    """Install a directory tree under share/<pkg>/<src_dir> keeping its layout."""
    out = []
    for root, _dirs, files in os.walk(src_dir):
        if files:
            out.append((os.path.join('share', package_name, root),
                        [os.path.join(root, f) for f in files]))
    return out


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ] + tree('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pai-student',
    maintainer_email='noreply@example.com',
    description='YOLO-based object search robot on ROS 2 + Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_detector = yolo_search_robot.yolo_detector_node:main',
            'search_controller = yolo_search_robot.search_controller_node:main',
            'keyboard_teleop = yolo_search_robot.keyboard_teleop_node:main',
            'demo_recorder = yolo_search_robot.demo_recorder_node:main',
        ],
    },
)
