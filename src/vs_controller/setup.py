from setuptools import find_packages, setup

package_name = 'vs_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages = [package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Optional: install launch files if you want
        # ('share/' + package_name + '/launch', ['launch/my.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kr2',
    maintainer_email='kr2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'MPC_Plot = vs_controller.MPC_Plot:main',
            'Wheels_Plot = vs_controller.Wheels_Plot:main',
            'CBFs_Plot = vs_controller.CBFs_Plot:main',
            'Tracking_Error_Plot = vs_controller.Tracking_Error_Plot:main',
            'Virtual_Structure_Node = vs_controller.Virtual_Structure_Node:main',
            'visualization = vs_controller.visualization:main',
        ],
    },
)
