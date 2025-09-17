import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'cave_explorer'

def generate_data_files(share_path, dir):
    """Installing nested directories seems very complicated in ROS2. This is from StackOverflow: https://stackoverflow.com/a/76159267"""
    data_files = []
    
    for path, _, files in os.walk(dir):
        list_entry = (share_path + path, [os.path.join(path, f) for f in files if not f.startswith('.')])
        data_files.append(list_entry)

    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*launch.[pxy][yma]*')),
        ('share/' + package_name + '/config', glob('config/*')),
    ] + generate_data_files('share/' + package_name + '/', 'urdf') + generate_data_files('share/' + package_name + '/', 'worlds'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Graeme Best',
    maintainer_email='graeme.best@uts.edu.au',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cave_explorer = cave_explorer.cave_explorer:main'
        ],
    },
)
