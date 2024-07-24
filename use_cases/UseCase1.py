import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
from utils.leagues import get_league_metadata

# Add the utils directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# Import the utility functions
from UseCase1_utils import calculate_score_probability, calculate_score_probabilities

def show():
    st.header('Use Case 1: Goals Scored')
    st.write('Goals scored probabilities for selected home and away teams.')
    
    # Get league metadata
    clubs, num_league_positions = get_league_metadata('epl')
    
    max_goal_count_home = 10
    max_goal_count_away = 10
    
    # Dropdowns for selecting home and away teams
    home_team = st.selectbox('Select Home Team:', clubs)
    away_team = st.selectbox('Select Away Team:', [team for team in clubs if team != home_team])
    
    # Dropdowns for selecting number of goals
    home_goals_options = [str(i) for i in range(max_goal_count_home + 1)]
    away_goals_options = [str(i) for i in range(max_goal_count_away + 1)]
    
    home_goals = st.selectbox('Select number of Home goals:', ['Select'] + home_goals_options)
    away_goals = st.selectbox('Select number of Away goals:', ['Select'] + away_goals_options)
    
    # Define a function to create a 3D plot
    def create_3d_plot():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0, max_goal_count_home + 1)
        y = np.arange(0, max_goal_count_away + 1)
        x, y = np.meshgrid(x, y)
        
        # Call the utility function to get probabilities
        z = calculate_score_probabilities(home_team, away_team, max_goal_count_home, max_goal_count_away)
        
        ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(z.flatten()), 1, 1, z.flatten(), color='forestgreen')
        ax.set_xlabel('Goals scored by ' + home_team, labelpad=-7, fontsize=8.5)
        ax.set_ylabel('Goals scored by ' + away_team, labelpad=-5, fontsize=8.5)
        ax.set_zlabel('Probability (%)', labelpad=-4, fontsize=8.5)
        
        # Adjust the tick labels
        ax.set_xticks(np.arange(0, max_goal_count_home + 1))
        ax.set_yticks(np.arange(0, max_goal_count_away + 1))

        # Set tick labels font size
        ax.tick_params(axis='x', labelsize=7, pad=-5)
        ax.tick_params(axis='y', labelsize=7, pad=-3)
        ax.tick_params(axis='z', labelsize=7, pad=-0.5)
        
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        st.pyplot(fig)
    
    # Check if goals are selected
    if home_goals == 'Select' or away_goals == 'Select':
        create_3d_plot()
    else:
        # Calculate probability
        probability = calculate_score_probability(home_team, away_team, int(home_goals), int(away_goals))
        st.markdown(f"""
            <div style='text-align: left;'>
                <span style='font-size: 24px; font-weight: bold; color: black;'>Probability:</span>
                <span style='font-size: 18px; color: green;'>{probability:.8f}</span>
            </div>
            """, unsafe_allow_html=True)

