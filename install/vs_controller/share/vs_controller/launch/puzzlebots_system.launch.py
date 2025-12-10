from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():

    def term(cmd, name):
        return ExecuteProcess(
            cmd=[
                "terminator", "--title", name, "--",
                "bash", "-c", cmd
            ],
            shell=False
        )

    # --- List of commands (one per terminal) ---
    cmds = [

        # Virtual Structure
        ("ros2 run vs_controller Virtual_Structure_Node", "Virtual Structure"),

        # Plots
        ("ros2 run vs_controller CBFs_Plot.py", "CBF Plot"),
        ("ros2 run vs_controller Tracking_Error_Plot.py", "Tracking Error Plot"),
        ("ros2 run vs_controller MPC_Plot.py", "MPC Plot"),

        # Visualization
        ("ros2 run vs_controller visualization.py", "Visualization"),
    ]

    # wrap them into terminal launch actions
    procs = [term(cmd, name) for (cmd, name) in cmds]

    return LaunchDescription(procs)
