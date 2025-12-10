import pandas as pd
import numpy as np

def generate_tasks(num_tasks: int, task_types: list) -> pd.DataFrame:
    
    tasks_data = []
    
    HUMAN_POSE_FEATURES = 45
    OBJECT_POS_FEATURES = 5
    
    for i in range(num_tasks):
        task_type = np.random.choice(task_types)
        
        human_pose = np.random.rand(HUMAN_POSE_FEATURES) 
        object_positions = np.random.rand(OBJECT_POS_FEATURES) 
        
        row_data = [task_type] + list(human_pose) + list(object_positions)
        tasks_data.append(row_data)
        
    columns = (
        ['task_type'] + 
        ['human_' + str(i) for i in range(HUMAN_POSE_FEATURES)] + 
        ['obj_' + str(i) for i in range(OBJECT_POS_FEATURES)]
    )
    
    df = pd.DataFrame(tasks_data, columns=columns)
    return df