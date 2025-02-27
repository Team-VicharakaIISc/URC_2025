from setuptools import find_packages, setup

package_name = 'nav2_goal_writer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pratham',
    maintainer_email='prathamgupta423@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local_goal_writer = nav2_goal_writer.ros2_nav2_goal_writer:main',
            'global_goal_writer = nav2_goal_writer.ros2_global_goal_writer:main',
            'mock_gnss_publisher = nav2_goal_writer.mock_gnss_pub:main',
            'ros2_local_search_goal_writer = nav2_goal_writer.ros2_local_search_goal_writer:main',
            'object_detection = nav2_goal_writer.object_detection:main',
            'aruco_detector = nav2_goal_writer.aruco_detector:main', 
        ],
    },
)
