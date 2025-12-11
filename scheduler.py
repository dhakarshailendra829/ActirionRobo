import pandas as pd
import numpy as np
import random
from typing import List

class AdvancedTaskScheduler:
    def __init__(self):
        self.HUMAN_POSE_FEATURES = 45
        self.OBJECT_POS_FEATURES = 5
        
        self.task_configs = {
            "pick": {"human_height": [1.6, 1.85], "object_height": [0.2, 1.2], "target_offset": [0.8, 1.5, 0.3], "priority": 3},
            "place": {"human_height": [1.4, 1.8], "object_height": [0.8, 1.6], "target_offset": [1.2, 2.0, 1.0], "priority": 4},
            "move": {"human_height": [1.5, 1.9], "object_height": [0.1, 0.8], "target_offset": [-1.5, 1.5, 0.5], "priority": 2},
            "sort": {"human_height": [1.55, 1.82], "object_height": [0.3, 1.0], "target_offset": [0.5, 1.2, 1.8], "priority": 3}
        }
    
    def generate_realistic_human_pose(self, task_type: str) -> np.ndarray:
        config = self.task_configs[task_type]
        human_pose = np.random.normal(0.5, 0.3, self.HUMAN_POSE_FEATURES)  
        
        torso_height = np.random.uniform(*config["human_height"])
        if 3 <= self.HUMAN_POSE_FEATURES:
            human_pose[0:3] = [torso_height, 0.0, 0.0]
            
        return np.clip(human_pose, -2, 2)
    
    def generate_object_features(self, task_type: str) -> np.ndarray:
        config = self.task_configs[task_type]
        obj_pos = np.random.normal(0.5, 0.3, self.OBJECT_POS_FEATURES)  
        
        obj_height = np.random.uniform(*config["object_height"])
        if 3 <= self.OBJECT_POS_FEATURES:
            obj_pos[2] = obj_height
            
        return np.clip(obj_pos, 0, 2)
    
    def generate_task_target(self, task_type: str) -> tuple:
        config = self.task_configs[task_type]
        base_offset = np.array(config["target_offset"])
        target = base_offset + np.random.normal(0, 0.3, 3)
        target[2] = np.clip(target[2], 0.2, 3.0)
        difficulty = np.linalg.norm(target)
        return target, difficulty
    
    def calculate_priority(self, task_type: str, difficulty: float) -> float:
        base_priority = self.task_configs[task_type]["priority"]
        return base_priority * max(0.5, 1.2 - difficulty * 0.1)
    
    def generate_tasks(self, num_tasks: int, task_types: List[str], difficulty_scale: float = 1.0) -> pd.DataFrame:
        tasks_data = []
        
        for i in range(num_tasks):
            task_type = np.random.choice(task_types)
            
            human_pose = self.generate_realistic_human_pose(task_type)
            obj_features = self.generate_object_features(task_type)
            target_pos, difficulty = self.generate_task_target(task_type)
            priority = self.calculate_priority(task_type, difficulty)
            
            row = (
                [task_type, i+1000, priority, difficulty] +
                human_pose.tolist() +
                obj_features.tolist() +
                target_pos.tolist()
            )
            tasks_data.append(row)
        
        columns = (
            ["task_type", "task_id", "priority", "difficulty"] +
            [f"human_{i}" for i in range(self.HUMAN_POSE_FEATURES)] +
            [f"obj_{i}" for i in range(self.OBJECT_POS_FEATURES)] +
            ["target_x", "target_y", "target_z"]
        )
        
        df = pd.DataFrame(tasks_data, columns=columns)
        return df.sort_values('priority', ascending=False).reset_index(drop=True)

def generate_tasks(num_tasks: int, task_types: list, difficulty_scale: float = 1.0) -> pd.DataFrame:
    scheduler = AdvancedTaskScheduler()
    return scheduler.generate_tasks(num_tasks, task_types, difficulty_scale)
