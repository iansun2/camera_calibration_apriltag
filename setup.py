#!/usr/bin/env python
from glob import glob
from setuptools import setup

PACKAGE_NAME = "camera_calibration_apriltag"

setup(
    name=PACKAGE_NAME,
    version='4.0.0',
    packages=[PACKAGE_NAME, PACKAGE_NAME + ".nodes"],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
        ('share/' + PACKAGE_NAME + '/launch', glob('launch/*.launch.py')),
    ],
    py_modules=[],
    package_dir={'': 'src'},
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    author='James Bowman, Patrick Mihelich',
    maintainer='Ian Sun',
    maintainer_email='iansun2004@gmail.com',
    keywords=['ROS2'],
    description='Camera calibration from AprilTag detections (AprilGrid target) with a PySide6 GUI.',
    license='BSD',
    tests_require=[
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'cameracalibrator = camera_calibration_apriltag.nodes.cameracalibrator:main',
        ],
    },
)
