# robot_env.py - FINAL VERSION: Delegates Kinematics to kinematics.py
import numpy as np
# Import the new kinematics module for calculating positions
from kinematics import forward_kinematics_3d 

class RobotEnv:
    """
    Simulates a simple multi-jointed robot arm environment.
    Joint angles represent the rotation applied at each link.
    """
    def __init__(self, num_joints=6, link_length=1.0):
        self.num_joints = num_joints
        # Assuming fixed link length for simplicity in visualization
        self.link_length = link_length 
        self.joint_angles = np.zeros(num_joints)
        self.trajectory = [] # Stores history of joint angle states

    def reset(self):
        """Resets the robot to its initial state (all joints at 0)."""
        self.joint_angles = np.zeros(self.num_joints)
        self.trajectory = []
        return self.joint_angles

    def step(self, action):
        """
        Updates the joint angles based on the AI predicted action.
        """
        # Directly sets the joint angles based on the action/prediction
        self.joint_angles = np.array(action)
        self.trajectory.append(self.joint_angles.copy()) 
        return self.joint_angles

    def get_link_positions(self, joint_angles=None):
        """
        Calculates the 3D coordinates (X, Y, Z) for the base and all joints/links.
        This function now delegates the calculation to the dedicated kinematics module.
        
        Returns:
            tuple: (x_coords, y_coords, z_coords) - lists of coordinates for all joints/links.
        """
        if joint_angles is None:
            joint_angles = self.joint_angles
            
        # --- DELEGATION to kinematics.py ---
        # The complex logic is moved out for better organization
        return forward_kinematics_3d(joint_angles, link_length=self.link_length)
    
    def get_end_effector_position(self, joint_angles=None):
        """
        Compatibility function, redirects to get_link_positions.
        """
        return self.get_link_positions(joint_angles)