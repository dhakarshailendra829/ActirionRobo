# app.py - FINAL CORRECTED VERSION: Fixed StreamlitDuplicateElementId + AUTOPLAY/ANIMATION
import sys, os
import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time # NEW: For animation delay

# Import updated local modules
from scheduler import generate_tasks
from robot_env import RobotEnv
from visualization import (
    plot_2d_robot,
    plot_3d_robot,
    plot_performance_graph,
    plot_end_effector_path,
    plot_performance_bar_3d
)

# NEW IMPORT for advanced functionality
from training_loop import train_imitation_model 

# --- Configuration & Model Setup ---
st.set_page_config(
    page_title="Advanced Smart Robot Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model setup (unchanged)
input_dim = 50
output_dim = 6
device = torch.device("cpu")

class ImitationModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

model = ImitationModel(input_dim, output_dim)
model_path = os.path.join("trained_models", "imitation_model.pt")

try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.markdown(f'<div style="background-color: #007700; color: white; padding: 5px; border-radius: 5px; font-size: 14px; text-align: center;">Trained imitation model loaded</div>', unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Model file not found. Using untrained model.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

model.eval()
robot = RobotEnv(num_joints=output_dim)
robot.reset()
# -------------------------

# --- SESSION STATE INITIALIZATION FOR ANIMATION ---
if 'autoplay_enabled' not in st.session_state:
    st.session_state.autoplay_enabled = False
if 'current_task_index' not in st.session_state:
    st.session_state.current_task_index = 0
if 'rotation_angle_y' not in st.session_state:
    st.session_state.rotation_angle_y = 1.5 # Initial camera rotation 
# ----------------------------------------------------


# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Task Scheduler Settings")
    
    # --- AUTOPLAY SWITCH ADDED HERE ---
    st.markdown("---")
    st.subheader("Animation Controls")
    st.session_state.autoplay_enabled = st.checkbox(
        "🔄 Enable Auto-Play (Animate Tasks)", 
        value=st.session_state.autoplay_enabled,
        key="autoplay_switch"
    )
    st.markdown("---") 
    
    num_tasks = st.number_input("Number of Tasks", min_value=1, max_value=20, value=5, label_visibility="collapsed")
    st.markdown("Select Task Types", unsafe_allow_html=True)
    task_types = st.multiselect(
        "Select Task Types", ["pick", "place", "move", "sort"],
        default=["pick", "place", "move", "sort"],
        label_visibility="collapsed"
    )
    
    st.markdown("---") 
    st.subheader("Manual Joint Control")
    joint_sliders = [st.slider(f"Joint {i}", -np.pi, np.pi, 0.0, key=f"manual_joint_{i}") for i in range(output_dim)]
    
    # Run button should reset animation state
    if st.button("🚀 Generate & Run Simulation", type="primary"):
        st.session_state.current_task_index = 0 # Reset animation index
        st.session_state.rotation_angle_y = 1.5 # Reset rotation
        # Set a flag to ensure simulation runs even if autoplay is off
        st.session_state['run_simulation_flag'] = True
    else:
        st.session_state['run_simulation_flag'] = False
# ------------------------------------

# --- Main Title and Description ---
st.title("🤖 Advanced Self-Learning Industrial Robot Simulator")
st.markdown("""
This simulator demonstrates an **AI-powered robotic arm** that learns and executes tasks. It includes:
* Intelligent task scheduling
* 2D & 3D trajectory visualization
* Manual joint control
* 3D path & performance analysis
""")

# --- Main Content: Tabs ---
tab_overview, tab_scheduling, tab_3d_viz, tab_data_analysis, tab_training = st.tabs([
    "Overview", 
    "Task Scheduling", 
    "3D Visualization", 
    "Data Analysis",
    "🧠 Training" 
])

# --- Scheduled Tasks Header ---
st.subheader("📋 Scheduled Tasks")

if 'tasks_df' not in st.session_state:
    st.session_state['tasks_df'] = pd.DataFrame(columns=['task_type'] + [f'human_{i}' for i in range(45)] + [f'obj_{i}' for i in range(5)])
if 'robot_actions' not in st.session_state:
    st.session_state['robot_actions'] = []

st.dataframe(st.session_state['tasks_df'], use_container_width=True)

# ----------------- Simulation Logic -----------------
# Trigger simulation if the button was pressed or if it was rerun by the animation logic
if st.session_state['run_simulation_flag']:
    # --- The original simulation calculation logic runs here ---
    tasks_df = generate_tasks(num_tasks, task_types)
    
    robot_actions = []
    end_effector_positions = []
    task_colors = []
    
    task_color_map = {
        "pick": "#FF66FF", 
        "place": "#33CCFF", 
        "move": "#66FF33", 
        "sort": "#FF3333" 
    }
    
    for idx, row in tasks_df.iterrows():
        features_list = [f'human_{i}' for i in range(45)] + [f'obj_{i}' for i in range(5)]
        features = row[features_list].values.astype(np.float32)
        
        with torch.no_grad():
            action = model(torch.from_numpy(features)).detach().numpy()
        
        robot_actions.append(action)
        
        # Calculate end effector position (simplified)
        x, y, z = 0, 0, 0
        angle = 0
        for i, a in enumerate(action):
            angle += a 
            x += 1.0 * np.cos(angle)
            y += 1.0 * np.sin(angle)
            z = i * 0.5 
        end_effector_positions.append((x, y, z))
        
        ttype = row['task_type'] if 'task_type' in row else 'move'
        task_colors.append(task_color_map.get(ttype, "gray"))

    st.session_state['tasks_df'] = tasks_df
    st.session_state['robot_actions'] = robot_actions
    st.session_state['end_effector_positions'] = end_effector_positions
    st.session_state['task_colors'] = task_colors
    
    st.toast("✅ Simulation Run Complete! Visualizations updated.")
    
# ----------------- TAB: Overview (With Animation Logic) -----------------
with tab_overview:
    
    col_3d_robot, col_3d_scatter = st.columns([1, 1])
    
    if st.session_state['robot_actions']:
        
        # Determine the action set to display based on autoplay status
        actions_list = st.session_state['robot_actions']
        num_actions = len(actions_list)
        
        if st.session_state.autoplay_enabled:
            # Display current task position
            display_index = st.session_state.current_task_index % num_actions
        else:
            # Display final position if autoplay is off, or the first task if index is 0
            display_index = num_actions - 1 

        current_action_set = [actions_list[display_index]]

        # --- Dynamic Rotation Update ---
        # Update rotation angle for the next frame
        if st.session_state.autoplay_enabled:
            st.session_state.rotation_angle_y = (st.session_state.rotation_angle_y + 0.05) % (2 * np.pi) 

        # --- 3D Robot Plot ---
        with col_3d_robot:
            st.markdown("### Intelligent Task Scheduling")
            # NOTE: We need to pass the rotation angle to visualization.py if we want the plot to spin.
            # Since plot_3d_robot doesn't accept a camera angle yet, this plot won't spin unless we update visualization.py again.
            # For now, we only use the animation to cycle through tasks.
            plot_3d_robot(
                current_action_set,
                output_dim,
                title=f"3D Robot Arm (Task {display_index + 1}/{num_actions})",
                end_effector_positions=st.session_state['end_effector_positions'],
                task_colors=st.session_state['task_colors'],
                key="overview_3d_robot_plot"
            )

        # --- Trajectory Plot ---
        with col_3d_scatter:
            st.markdown("### 2D & 3D Trajectory Visualization")
            plot_end_effector_path(
                st.session_state['end_effector_positions'], 
                st.session_state['task_colors'],
                key="overview_trajectory_plot" 
            )
            
        # --- AUTOPLAY LOOP ---
        if st.session_state.autoplay_enabled:
            # Move to the next index
            next_index = (st.session_state.current_task_index + 1) % num_actions
            st.session_state.current_task_index = next_index

            # Pause and Rerun (Creates the animation effect)
            time.sleep(0.5) # Animation speed (0.5 seconds per task)
            st.rerun() 
            
    else:
        st.info("Run the simulation to generate the 3D visualizations.")

# ----------------- Other Tabs (Unchanged logic) -----------------
with tab_scheduling:
    st.header("📋 AI Task Scheduling & Predicted Actions")
    
    if st.session_state['robot_actions']:
        st.subheader("Predicted Robot Actions (Joint Angles)")
        actions_df = pd.DataFrame(st.session_state['robot_actions'], columns=[f'Action_Joint_{i}' for i in range(output_dim)])
        st.dataframe(actions_df, use_container_width=True)
    else:
        st.info("Run the simulation using the sidebar button to see predicted actions.")


with tab_3d_viz:
    st.header("👁️ Detailed 3D Trajectory Analysis")
    
    if st.session_state['robot_actions']:
        plot_end_effector_path(
            st.session_state['end_effector_positions'], 
            st.session_state['task_colors'],
            key="detailed_trajectory_plot" 
        )
    else:
        st.info("Run the simulation to generate detailed 3D trajectory plots.")
        
with tab_data_analysis:
    st.header("📈 Performance & Data Analysis")
    
    if st.session_state['robot_actions']:
        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown("**3D Performance Bar Chart (Action vs Task Type)**")
            plot_performance_bar_3d(st.session_state['robot_actions'], st.session_state['tasks_df'], key="performance_bar_3d") 
            
        with colB:
            st.markdown("**Action Magnitude per Task (Action Intensity)**")
            plot_performance_graph(st.session_state['robot_actions'], key="action_magnitude_line") 
            
    else:
        st.info("Run the simulation to generate performance analysis charts.")
        
with tab_training:
    st.header("🧠 AI Model Training & Optimization")
    
    if 'tasks_df' not in st.session_state or st.session_state['tasks_df'].empty:
        st.warning("⚠️ Run the simulation first to generate data for training.")
    else:
        st.subheader("Training Parameters")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, key="training_epochs")
        with col_param2:
            lr = st.number_input("Learning Rate", min_value=1e-5, max_value=0.1, value=0.001, format="%e", key="training_lr")
        
        train_button = st.button("🔴 Start Training", type="secondary")
        
        if train_button:
            with st.spinner(f"Training model for {epochs} epochs..."):
                loss_history = train_imitation_model(model, st.session_state['tasks_df'].copy(), epochs, lr)
            
            st.success("✅ Training Complete!")
            
            st.subheader("Training Loss History")
            fig_loss = go.Figure(data=[go.Scatter(y=loss_history, mode='lines+markers', name='Loss', 
                                                 line=dict(color='#FF66FF', width=4))])
            fig_loss.update_layout(
                title="Model Training Loss (MSE)",
                xaxis_title="Epoch",
                yaxis_title="Loss Value",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True, key="training_loss_chart")