import numpy as np

def forward_kinematics_3d(joint_angles, link_length=1.0):
    x_coords, y_coords, z_coords = [0.0], [0.0], [0.0]
    cumulative_angle = 0.0
    current_z = 0.0 

    for i, angle_change in enumerate(joint_angles):
        cumulative_angle += angle_change
        
        dx = link_length * np.cos(cumulative_angle)
        dy = link_length * np.sin(cumulative_angle)
        
        dz = 0.5 
        current_z += dz
        
        x_coords.append(x_coords[-1] + dx)
        y_coords.append(y_coords[-1] + dy)
        z_coords.append(current_z)
            
    return x_coords, y_coords, z_coords

def inverse_kinematics_simple(target_xyz, num_joints):
    target_magnitude = np.sqrt(target_xyz[0]**2 + target_xyz[1]**2 + target_xyz[2]**2)
    
    angles = np.ones(num_joints) * (target_magnitude / num_joints / 5.0)
    
    angles += np.random.uniform(-0.1, 0.1, num_joints)
    
    return angles

from kinematics import forward_kinematics_3d

class RobotEnv:
    def get_link_positions(self, joint_angles=None):
        
        if joint_angles is None:
            joint_angles = self.joint_angles
            
        return forward_kinematics_3d(joint_angles, link_length=self.link_length)
    
    def get_end_effector_position(self, joint_angles=None):
        return self.get_link_positions(joint_angles)