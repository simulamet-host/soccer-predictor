import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def show():
    st.header('Score prediction probability')
    st.write('Here you can calculate probabilities of different football results between home and away team.')
    
    # List of football clubs
    clubs = [
        'Arsenal', 'Aston Villa', 'Blackburn Rovers', 'Chelsea', 'Coventry City', 
        'Crystal Palace', 'Everton', 'Ipswich Town', 'Leeds United', 'Liverpool', 
        'Manchester City', 'Manchester United', 'Middlesbrough', 'Norwich City', 
        'Nottingham Forest', 'Oldham Athletic', 'Queens Park Rangers', 
        'Sheffield United', 'Sheffield Wednesday', 'Southampton', 'Tottenham Hotspur', 
        'Wimbledon'
    ]
    
    max_goal_count_home = 10
    max_goal_count_away = 10
    
    # Dropdowns for selecting home and away teams
    home_team = st.selectbox('Select Home team', clubs)
    away_team = st.selectbox('Select Away team', [team for team in clubs if team != home_team])
    
    # Dropdowns for selecting number of goals
    home_goals_options = [str(i) for i in range(max_goal_count_home + 1)]
    away_goals_options = [str(i) for i in range(max_goal_count_away + 1)]
    
    home_goals = st.selectbox('Select number of Home goals:', ['Select'] + home_goals_options)
    away_goals = st.selectbox('Select number of away goals:', ['Select'] + away_goals_options)
    
    # Define a function to create a 3D plot
    def create_3d_plot():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0, max_goal_count_home + 1)
        y = np.arange(0, max_goal_count_away + 1)
        x, y = np.meshgrid(x, y)
        z = np.random.rand(max_goal_count_away + 1, max_goal_count_home + 1) * 20
        
        ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(z.flatten()), 1, 1, z.flatten(), color='forestgreen')
        ax.set_xlabel('Goals scored from ' + home_team, labelpad=-7, fontsize=9)
        ax.set_ylabel('Goals scored from ' + away_team, labelpad=-5, fontsize=9)
        ax.set_zlabel('Probability score in percentage', labelpad=-3, fontsize=9)
        
        # Adjust the tick labels
        ax.set_xticks(np.arange(0, max_goal_count_home + 1))
        ax.set_yticks(np.arange(0, max_goal_count_away + 1))

        # Set tick labels font size
        ax.tick_params(axis='x', labelsize=8, pad=-5)
        ax.tick_params(axis='y', labelsize=8, pad=-3)
        ax.tick_params(axis='z', labelsize=8, pad=0)
        
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        st.pyplot(fig)
    
    # Check if goals are selected
    if home_goals == 'Select' or away_goals == 'Select':
        create_3d_plot()
    else:
        # Calculate probability
        probability = np.random.rand()
        st.write(f"Probability: {probability}")

