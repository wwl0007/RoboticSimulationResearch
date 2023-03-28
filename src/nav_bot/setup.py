# Author: Wesley Lowman
# Mentor: Dr. Vahid Azimi
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
from setuptools import setup
from glob import glob
import os

package_name = 'nav_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob('launch/*')),
        (os.path.join('share', package_name,'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name,'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name,'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wes',
    maintainer_email='wwl0007@auburn.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drive_ugv = nav_bot.drive_ugv:main' ,
            'find_goal = nav_bot.find_goal:main',
            'video_saver = nav_bot.video_saver:main',
            'navigator = nav_bot.navigator:main',
        ],
    },
)
