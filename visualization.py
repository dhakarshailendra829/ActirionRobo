import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd

PLOTLY_CONFIG = {
    'displayModeBar': False,
    'responsive': True,
    'staticPlot': False
}

DARK_BG = '#0B0F1A'
GRID_COLOR = "#0A2E99"
NEON_BLUE = "#0CE5F1"
NEON_PURPLE = '#AF2AFF'
NEON_ORANGE = '#FF7F0E'
ROBOT_LINK_COLOR = "#F0590D"
BASE_COLOR = '#5A3F7B'
TASK_COLORS = {
    "pick": "#EC17EC", "place": "#33CCFF", 
    "move": "#66FF33", "sort": "#FF3333"
}

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
        current_z += 0.6
        x_coords.append(x_coords[-1] + dx)
        y_coords.append(y_coords[-1] + dy)
        z_coords.append(current_z)
    return x_coords, y_coords, z_coords

def _make_holo_cylinder(radius, z_top, n=60):
    theta = np.linspace(0, 2*np.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, z_top)
    return x, y, z

def _create_animated_robot_arm(actions, current_frame=0, num_joints=6):
    if not actions:
        actions = [[0.2, -0.5, 0.3, 0.1][:num_joints]]
    
    current_action = actions[min(current_frame, len(actions)-1)]
    x_coords, y_coords, z_coords = _get_link_positions_3d(current_action)
    
    traces = []
    
    radius = max(len(current_action) + 1.5, 3.5)
    z_max = len(current_action) * 0.6 + 1.5
    theta = np.linspace(0, 2*np.pi, 80)
    x_cyl = radius * np.cos(theta)
    y_cyl = radius * np.sin(theta)
    z_cyl = np.zeros_like(theta)
    
    traces.append(go.Mesh3d(
        x=np.append(x_cyl, 0), y=np.append(y_cyl, 0), z=np.append(z_cyl, 0),
        alphahull=5, opacity=0.25, color=BASE_COLOR,
        name='Base', hoverinfo='none', showlegend=False
    ))
    
    x_ring, y_ring, z_ring = _make_holo_cylinder(radius*0.95, z_top=z_max, n=80)
    traces.append(go.Scatter3d(x=x_ring, y=y_ring, z=z_ring, mode='lines',
                              line=dict(color=NEON_BLUE, width=3),
                              showlegend=False, hoverinfo='none'))
    
    x_ring2, y_ring2, z_ring2 = _make_holo_cylinder(radius*0.95, z_top=0.1, n=80)
    traces.append(go.Scatter3d(x=x_ring2, y=y_ring2, z=z_ring2, mode='lines',
                              line=dict(color=NEON_PURPLE, width=3),
                              showlegend=False, hoverinfo='none'))
    
    traces.append(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines', line=dict(color=ROBOT_LINK_COLOR, width=22),
        hoverinfo='none', name='Robot Arm'
    ))
    
    pulse_colors = ['#FFAA00', '#FFCC33', '#FFFF66', '#CCFF33', '#66FF99', '#33FFCC']
    traces.append(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=[4,6,8,10,12,16][min(len(x_coords)-1,5)], 
                   color=pulse_colors[:len(x_coords)], 
                   symbol='circle', line=dict(width=2, color='white')),
        hoverinfo='text', text=[f'Joint {i}' for i in range(len(x_coords))],
        showlegend=False
    ))
    
    end_x, end_y, end_z = x_coords[-1], y_coords[-1], z_coords[-1]
    traces.append(go.Scatter3d(
        x=[end_x], y=[end_y], z=[end_z],
        mode='markers',
        marker=dict(size=20, color=NEON_BLUE, symbol='diamond', line=dict(width=3, color='black')),
        name='End Effector', hoverinfo='text', text=['End Effector']
    ))
    
    return traces, (end_x, end_y, end_z)

def _create_trajectory_with_current_highlight(positions, task_colors, current_frame=0, tasks_df=None):
    if not positions:
        positions = [(i*0.8, np.sin(i*0.6)*0.6, i*0.4) for i in range(6)]
    
    xx, yy, zz = zip(*positions)
    
    traces = [go.Scatter3d(
        x=xx, y=yy, z=zz, mode='lines',
        line=dict(color='rgba(17,245,255,0.3)', width=4),
        name='Full Path', showlegend=False, hoverinfo='none'
    )]
    
    if current_frame < len(positions):
        cx, cy, cz = positions[current_frame]
        traces.append(go.Scatter3d(
            x=[cx], y=[cy], z=[cz], mode='markers+text',
            marker=dict(size=25, color='gold', symbol='diamond', line=dict(width=4, color='orange')),
            text=[f'CURRENT\nTask {current_frame+1}'],
            textposition="middle center",
            textfont=dict(color='white', size=12),
            name='Current Task', hoverinfo='text'
        ))
    
    cube_size = 0.28
    for i, (cx, cy, cz) in enumerate(positions):
        if i > current_frame + 1: continue
        
        color = TASK_COLORS.get(tasks_df.iloc[i]['task_type'], '#888888') if tasks_df is not None and i < len(tasks_df) else '#666666'
        opacity = 1.0 if i <= current_frame else 0.6
        
        s = cube_size / 2.0
        xv = [cx - s, cx + s, cx + s, cx - s, cx - s, cx + s, cx + s, cx - s]
        yv = [cy - s, cy - s, cy + s, cy + s, cy - s, cy - s, cy + s, cy + s]
        zv = [cz - s, cz - s, cz - s, cz - s, cz + s, cz + s, cz + s, cz + s]
        
        try:
            traces.append(go.Mesh3d(
                x=xv, y=yv, z=zv, color=color, opacity=opacity, 
                alphahull=0, hoverinfo='text', text=[f'Task {i+1}'],
                name=f'Task {i+1}' if i == current_frame else None, showlegend=i==0
            ))
        except:
            traces.append(go.Scatter3d(
                x=[cx], y=[cy], z=[cz], mode='markers',
                marker=dict(size=16 if i==current_frame else 12, color=color, symbol='square'),
                hoverinfo='text', text=[f'Task {i+1}']
            ))
    
    return traces

def combined_visualization(
    robot_actions,
    end_effector_positions,
    tasks_df=None,
    num_joints=6,
    current_frame=0,
    title_left="Live Robot Arm",
    title_right="Task Trajectory",
    key=None
):
    fig = make_subplots(rows=1, cols=2,
                       specs=[[{"type": "scene"}, {"type": "scene"}]],
                       horizontal_spacing=0.06,
                       subplot_titles=[title_left, title_right])

    robot_traces, end_pos = _create_animated_robot_arm(robot_actions, current_frame, num_joints)
    for trace in robot_traces:
        fig.add_trace(trace, row=1, col=1)

    traj_traces = _create_trajectory_with_current_highlight(
        end_effector_positions, [], current_frame, tasks_df
    )
    for trace in traj_traces:
        fig.add_trace(trace, row=1, col=2)

    total_frames = max(len(robot_actions), len(end_effector_positions), 1)
    frame_info = f"Frame {current_frame+1}/{total_frames}"
    
    scene_common = dict(
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=NEON_PURPLE, showbackground=True, 
                  backgroundcolor=DARK_BG, showspikes=False, nticks=6, title=''),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=NEON_PURPLE, showbackground=True, 
                  backgroundcolor=DARK_BG, showspikes=False, nticks=6, title=''),
        zaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=NEON_PURPLE, showbackground=True, 
                  backgroundcolor=DARK_BG, showspikes=False, nticks=6, title=''),
        bgcolor=DARK_BG,
    )

    fig.update_layout(
        scene=dict(camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.45, y=1.45, z=0.6)),
                  aspectmode='cube', **scene_common),
        scene2=dict(camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.8, y=1.8, z=0.9)),
                   aspectmode='auto', **scene_common),
        title=dict(
            text=f"<b>{title_left}</b> &nbsp; {frame_info} &nbsp;&nbsp;&nbsp;&nbsp; <b>{title_right}</b>", 
            x=0.01, xanchor='left', font=dict(size=16, color='white')
        ),
        template='plotly_dark',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=10, r=10, t=80, b=10),
        showlegend=True,
        height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    status_text = f"Task {current_frame+1}/{total_frames} Active"
    fig.add_annotation(
        text=status_text, x=0.5, y=0.02, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color=NEON_BLUE),
        bgcolor='rgba(0,0,0,0.5)', bordercolor=NEON_BLUE, borderwidth=1
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Animated Visualization Demo")
    st.title("Live Animation Demo")
    
    demo_actions = [[0.1*i, -0.2*i, 0.3*i, 0.1*i][:6] for i in range(10)]
    demo_positions = [(i*0.3, np.sin(i*0.4)*0.5, i*0.2) for i in range(10)]
    
    frame_slider = st.slider("Demo Frame", 0, 9, 5)
    combined_visualization(demo_actions, demo_positions, current_frame=frame_slider)
