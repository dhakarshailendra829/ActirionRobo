# scheduler.py - FINAL VERSION: Random Task Generation
import pandas as pd
import numpy as np

def generate_tasks(num_tasks: int, task_types: list) -> pd.DataFrame:
    """
    Generate random tasks for the robot simulation. 
    Each task includes a type, simulated human pose data (45 features), 
    and object position data (5 features).
    
    Args:
        num_tasks (int): The number of tasks to generate.
        task_types (list): List of possible task strings (e.g., ['pick', 'place']).
        
    Returns:
        pd.DataFrame: A DataFrame containing all simulated task data.
    """
    tasks_data = []
    
    # Define feature counts (Consistent with model input_dim = 50)
    HUMAN_POSE_FEATURES = 45
    OBJECT_POS_FEATURES = 5
    
    for i in range(num_tasks):
        # 1. Randomly select a task type
        task_type = np.random.choice(task_types)
        
        # 2. Generate random features (e.g., normalized sensor data between 0 and 1)
        human_pose = np.random.rand(HUMAN_POSE_FEATURES) 
        object_positions = np.random.rand(OBJECT_POS_FEATURES) 
        
        # 3. Combine task type and features into a single row
        row_data = [task_type] + list(human_pose) + list(object_positions)
        tasks_data.append(row_data)
        
    # Define DataFrame columns for clarity
    columns = (
        ['task_type'] + 
        ['human_' + str(i) for i in range(HUMAN_POSE_FEATURES)] + 
        ['obj_' + str(i) for i in range(OBJECT_POS_FEATURES)]
    )
    
    # Create the final DataFrame
    df = pd.DataFrame(tasks_data, columns=columns)
    return df