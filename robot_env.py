import numpy as np
import time
import psutil
import random

class RobotEnv:    
    def __init__(self, num_joints=6, link_length=1.0):
        self.num_joints = num_joints
        self.link_length = link_length
        self.joint_angles = np.zeros(num_joints)
        self.joint_velocities = np.zeros(num_joints)  
        self.trajectory = []
        self.performance_log = []  
        
        self.max_torque = np.pi / 2  
        self.friction = 0.95  
        self.target_precision = 0.1  
        
        self.task_targets = []
        self.current_target = None
        
        self.total_steps = 0
        self.successful_tasks = 0
        
    def reset(self, task_target=None):
        self.joint_angles = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.trajectory = []
        self.performance_log = []
        self.total_steps = 0
        
        if task_target is not None:
            self.current_target = np.array(task_target)
        else:
            self.current_target = np.array([
                random.uniform(-2, 2),
                random.uniform(-1, 1),
                random.uniform(1, 3)
            ])
            
        self.task_targets.append(self.current_target.copy())
        return self.joint_angles, {"target": self.current_target}
    
    def step(self, action, task_target=None):
        self.total_steps += 1
        action = np.clip(np.array(action), -self.max_torque, self.max_torque)
    
        self.joint_velocities = (action - self.joint_angles) * 0.8 + self.joint_velocities * self.friction
        self.joint_angles += self.joint_velocities
    
        current_pose = self.joint_angles.copy()
        self.trajectory.append(current_pose)
    
        ee_pos = self.get_end_effector_position()
    
        if task_target is None:
            ee_error = 0.5  
            task_success = 0.8  
        else:
            ee_error = np.linalg.norm(ee_pos - task_target)
            task_success = max(0.0, 1.0 - (ee_error / 0.5))  
    
        cpu_usage = psutil.cpu_percent(interval=0.01)
        action_norm = np.linalg.norm(action)
        energy_used = np.sum(np.abs(self.joint_velocities))
    
        perf_data = {
        'timestamp': time.time(),
        'step': self.total_steps,
        'cpu_usage': cpu_usage,
        'action_norm': action_norm,
        'ee_error': ee_error,
        'energy_used': energy_used,
        'ee_position': ee_pos,
        'task_success': task_success,  
        'success_rate': task_success
        }
        self.performance_log.append(perf_data)
    
        reward = -ee_error - 0.1 * action_norm + 10.0 * task_success
    
        info = {
        'position': ee_pos,
        'velocity': self.joint_velocities,
        'error': ee_error,
        'cpu_usage': cpu_usage,
        'success': task_success,
        'success_rate': task_success,  
        'energy': energy_used,
        'target': task_target
        }
    
        return self.joint_angles, reward, False, info

    def get_link_positions(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.joint_angles
            
        x, y, z = [0.0], [0.0], [0.0]
        cumulative_angle = 0.0
        current_z = 0.0
        
        for angle in joint_angles:
            cumulative_angle += angle
            dx = self.link_length * np.cos(cumulative_angle)
            dy = self.link_length * np.sin(cumulative_angle)
            current_z += 0.6  
            
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
            z.append(current_z)
            
        return list(zip(x, y, z))
    
    def get_end_effector_position(self, joint_angles=None):
        positions = self.get_link_positions(joint_angles)
        return np.array(positions[-1])
    
    def get_performance_summary(self):
        if not self.performance_log:
            return {}
            
        log = self.performance_log
        return {
            'avg_cpu': np.mean([m['cpu_usage'] for m in log]),
            'max_error': np.max([m['ee_error'] for m in log]),
            'avg_action_norm': np.mean([m['action_norm'] for m in log]),
            'total_energy': np.sum([m['energy_used'] for m in log]),
            'success_rate': np.mean([m['task_success'] for m in log]),
            'total_steps': self.total_steps,
            'smoothness': 1.0 / (1.0 + np.std([m['action_norm'] for m in log])),
            'trajectory_length': len(self.trajectory)
        }
    
    def get_trajectory_data(self):
        return [self.get_end_effector_position(pose) for pose in self.trajectory]
    
    def set_task_target(self, target_pos):
        self.current_target = np.array(target_pos)
        self.task_targets.append(self.current_target.copy())
