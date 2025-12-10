import numpy as np
from kinematics import forward_kinematics_3d 

class RobotEnv:
    def __init__(self, num_joints=6, link_length=1.0):
        self.num_joints = num_joints
        self.link_length = link_length 
        self.joint_angles = np.zeros(num_joints)
        self.trajectory = [] 

    def reset(self):
        self.joint_angles = np.zeros(self.num_joints)
        self.trajectory = []
        return self.joint_angles

    def step(self, action):
        
        self.joint_angles = np.array(action)
        self.trajectory.append(self.joint_angles.copy()) 
        return self.joint_angles

    def get_link_positions(self, joint_angles=None):
        
        if joint_angles is None:
            joint_angles = self.joint_angles
            
        return forward_kinematics_3d(joint_angles, link_length=self.link_length)
    
    def get_end_effector_position(self, joint_angles=None):
        return self.get_link_positions(joint_angles)