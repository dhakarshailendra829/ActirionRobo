# visualization.py - FINAL LOOK MATCH: CYBERPUNK/HOLOGRAM STYLE
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# --- Global Plotly Styling Configuration and NEON Colors ---
PLOTLY_CONFIG = {
    'displayModeBar': False,
    'responsive': True
}
# Colors adjusted to match the image's specific hues (Deep Purple/Orange on Dark)
DARK_BG = '#0B0F1A'             
GRID_COLOR = '#1E253A'          
NEON_BLUE = '#11F5FF'           # End Effector / Holo lines
NEON_PURPLE = '#AF2AFF'         # Holo Accent / Points
NEON_ORANGE = '#FF7F0E'         # Joint / Robot Base Accent
ROBOT_LINK_COLOR = '#FF9966'    # The main orange-pink link color in the image
BASE_COLOR = '#5A3F7B'          # Dark Purple Base/Container Color

# --- Helper function for 3D Kinematics (Ensures consistency) ---
def _get_link_positions_3d(action, link_length=1.0):
    x_coords = [0.0]
    y_coords = [0.0]
    z_coords = [0.0]
    cumulative_angle = 0.0
    current_z = 0.0 
    
    for i, angle_change in enumerate(action):
        cumulative_angle += angle_change
        
        dx = link_length * np.cos(cumulative_angle)
        dy = link_length * np.sin(cumulative_angle)
        
        current_z += 0.5 
        
        x_coords.append(x_coords[-1] + dx)
        y_coords.append(y_coords[-1] + dy)
        z_coords.append(current_z) 
            
    return x_coords, y_coords, z_coords

# 1. plot_3d_robot (Cylinder Container Look)
def plot_3d_robot(
    robot_actions,
    num_joints,
    title="3D Robot Arm (Final State)", 
    end_effector_positions=None,
    task_colors=None,
    key=None 
):
    if not robot_actions:
        st.info("No robot actions to display.")
        return

    final_action = robot_actions[0] 
    x_coords, y_coords, z_coords = _get_link_positions_3d(final_action)
    
    fig3 = go.Figure()

    # --- 1. Holo-Cylinder Container (To match the image's base and surrounding glow) ---
    z_max = num_joints * 0.5 + 1.5
    radius = num_joints + 1.5
    
    # Generate points for the cylinder walls
    theta = np.linspace(0, 2 * np.pi, 50)
    x_cyl = radius * np.cos(theta)
    y_cyl = radius * np.sin(theta)
    
    # Base Disk (A subtle plane near the bottom)
    fig3.add_trace(go.Mesh3d(
        x=np.append(x_cyl, 0), y=np.append(y_cyl, 0), z=[0]*50 + [0],
        alphahull=5,
        opacity=0.3, # Dark transparent floor
        color=BASE_COLOR,
        hoverinfo='none', name='Base'
    ))
    
    # Holo Rings (Neon Blue/Purple rings)
    fig3.add_trace(go.Scatter3d(
        x=x_cyl, y=y_cyl, z=[z_max]*50, 
        mode='lines',
        line=dict(color=NEON_BLUE, width=3),
        opacity=0.6,
        hoverinfo='none', name='Holo Top'
    ))
    fig3.add_trace(go.Scatter3d(
        x=x_cyl, y=y_cyl, z=[0.1]*50, 
        mode='lines',
        line=dict(color=NEON_PURPLE, width=3),
        opacity=0.6,
        hoverinfo='none', name='Holo Base Ring'
    ))
    
    # --- 2. Plot Robot Links (To match the orange-pink link color) ---
    fig3.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines',
        line=dict(color=ROBOT_LINK_COLOR, width=20), # Wide links
        name='Robot Links',
        hoverinfo='none'
    ))

    # --- 3. Plot Joints and End-Effector (Neon Accents) ---
    fig3.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=10, color=NEON_ORANGE, symbol='circle', line=dict(width=2, color='white')),
        name='Joints',
        hoverinfo='text',
        text=[f'Joint {i}' for i in range(len(x_coords))]
    ))

    fig3.add_trace(go.Scatter3d(
        x=[x_coords[-1]], y=[y_coords[-1]], z=[z_coords[-1]],
        mode='markers',
        marker=dict(size=18, color=NEON_BLUE, symbol='diamond', line=dict(width=3, color='black')), 
        name='End Effector',
    ))
    
    # --- 4. Layout and Cyberpunk Scene ---
    fig3.update_layout(
        title=f'**{title}**',
        scene=dict(
            xaxis=dict(range=[-radius, radius], backgroundcolor=DARK_BG, gridcolor=GRID_COLOR, zerolinecolor=NEON_BLUE, showbackground=True, showspikes=False, nticks=num_joints*2),
            yaxis=dict(range=[-radius, radius], backgroundcolor=DARK_BG, gridcolor=GRID_COLOR, zerolinecolor=NEON_BLUE, showbackground=True, showspikes=False, nticks=num_joints*2),
            zaxis=dict(range=[0, z_max], backgroundcolor=DARK_BG, gridcolor=GRID_COLOR, zerolinecolor=NEON_BLUE, showbackground=True, showspikes=False, nticks=num_joints*2),
            
            aspectratio=dict(x=1, y=1, z=1.2), 
            bgcolor=DARK_BG,
            
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.5, y=1.5, z=0.5) 
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=450, 
        showlegend=False,
        template="plotly_dark"
    )

    st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG, key=key)


# 2. plot_end_effector_path (Grid Trajectory Look)
def plot_end_effector_path(positions, task_colors, key=None): 
    x, y, z = zip(*positions)
    fig = go.Figure()
    
    # 1. Path Line (Thicker Neon Glow)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=NEON_BLUE, width=8), 
        name='Trajectory Path',
    ))
    
    # 2. Task Points (Markers) with color coding
    # Using specific colors to match the points (e.g., Pink/Green/Orange/Blue)
    # The image uses large, distinct markers for start/end points.
    
    # Create distinct colors for the trajectory points based on index for a colorful look
    path_colors_list = [NEON_PURPLE, NEON_ORANGE, NEON_BLUE, '#33FF66'] * (len(positions) // 4 + 1)
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=10, 
            color=path_colors_list[:len(positions)], 
            opacity=1.0,
            symbol='circle', 
            line=dict(width=2, color='white')
        ),
        name='Task Points',
        hoverinfo='text'
    ))
    
    # --- Layout and Cyberpunk Scene ---
    fig.update_layout(
        title="**3D End-Effector Task Trajectory**", 
        scene=dict(
            aspectmode="cube",
            bgcolor=DARK_BG,
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            
            # Make Grid Denser and Neon, similar to the image's right plot
            xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=NEON_PURPLE, nticks=8, showbackground=True, backgroundcolor=DARK_BG, showspikes=False),
            yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=NEON_PURPLE, nticks=8, showbackground=True, backgroundcolor=DARK_BG, showspikes=False),
            zaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=NEON_PURPLE, nticks=8, showbackground=True, backgroundcolor=DARK_BG, showspikes=False),
            
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.8, y=1.8, z=0.8) # Adjusted for better angled view
            )
        ),
        template="plotly_dark",
        height=450, 
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


# 3. plot_2d_robot (Matplotlib)
def plot_2d_robot(action, num_joints, color=NEON_PURPLE, title="Current Joint Position Visualization"):
    # ... (Matplotlib code - unchanged) ...
    import matplotlib.pyplot as plt
    
    x, y = [0], [0]
    angle = 0
    link_length = 1.0
    for a in action:
        angle += a
        x.append(x[-1] + link_length * np.cos(angle))
        y.append(y[-1] + link_length * np.sin(angle))
    
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
    ax.plot(x, y, marker='o', linewidth=5, color=color, markersize=8, markeredgecolor='white', zorder=2)
    ax.scatter(x[-1], y[-1], color=NEON_ORANGE, s=150, zorder=3) 
    
    limit = max(num_joints * link_length + 1, 5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, color=color)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors='white')
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
    
    st.pyplot(fig) 


# 4. plot_performance_graph (Neon Line Graph)
def plot_performance_graph(robot_actions, title="📊 Joint Action Magnitude per Task", key=None): 
    # ... (Unchanged from previous fix) ...
    if not robot_actions:
        st.info("No actions data available for performance graph.")
        return
        
    actions_df = pd.DataFrame(robot_actions, columns=[f'Joint {i}' for i in range(len(robot_actions[0]))])
    tasks = actions_df.index.tolist()
    
    fig = go.Figure()
    
    for i, col in enumerate(actions_df.columns):
        fig.add_trace(go.Scatter(
            x=tasks, 
            y=actions_df[col].abs(), 
            mode='lines', 
            name=col,
            line=dict(width=3),
            opacity=0.9
        ))

    fig.update_layout(
        title=f"**{title}**",
        xaxis_title="Task Index",
        yaxis_title="Absolute Action Magnitude",
        template="plotly_dark",
        hovermode="x unified",
        height=400,
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key) 


# 5. plot_performance_bar_3d (Neon Bar Chart)
def plot_performance_bar_3d(robot_actions, tasks_df, key=None): 
    # ... (Unchanged from previous fix) ...
    if not robot_actions or tasks_df.empty:
        st.info("No data for 3D performance bar chart.")
        return
        
    actions_df = pd.DataFrame(robot_actions, columns=[f'J{i}' for i in range(len(robot_actions[0]))])
    
    df_combined = pd.concat([tasks_df['task_type'].reset_index(drop=True), actions_df], axis=1)
    df_grouped = df_combined.groupby('task_type').apply(lambda x: x.iloc[:, 1:].abs().mean()).reset_index()
    
    task_types = df_grouped['task_type'].tolist()
    joint_indices = [f'J{i}' for i in range(len(robot_actions[0]))]
    
    data = []
    
    BAR_COLORS = ['#33CCFF', '#FF66FF', '#66FF33', '#FF3333'] 
    
    for i, joint in enumerate(joint_indices):
        data.append(go.Bar(
            x=task_types,
            y=df_grouped[joint],
            name=joint,
            marker_color=BAR_COLORS, 
            opacity=0.9,
            hovertemplate = f"<b>{joint}</b><br>Task Type: %{{x}}<br>Avg Magnitude: %{{y:.3f}}<extra></extra>"
        ))
        
    fig = go.Figure(data=data)

    fig.update_layout(
        title='**📈 Average Joint Action Magnitude by Task Type**',
        barmode='group',
        scene=dict(
            xaxis_title='Task Type',
            yaxis_title='Average Action Magnitude',
            zaxis_title='Joint',
            bgcolor=DARK_BG,
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR),
            zaxis=dict(gridcolor=GRID_COLOR)
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)