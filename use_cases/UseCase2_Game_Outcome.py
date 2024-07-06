import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the utils directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# Import the utility function
from UseCase2 import calculate_win_lose_draw

def show():
    st.header('Use Case 2: Game Outcome')
    st.write('Game outcome (win-lose-draw) probabilities for selected home and away teams.')
    
    # List of football clubs
    clubs = [
        'Arsenal', 'Aston Villa', 'Blackburn Rovers', 'Chelsea', 'Coventry City', 
        'Crystal Palace', 'Everton', 'Ipswich Town', 'Leeds United', 'Liverpool', 
        'Manchester City', 'Manchester United', 'Middlesbrough', 'Norwich City', 
        'Nottingham Forest', 'Oldham Athletic', 'Queens Park Rangers', 
        'Sheffield United', 'Sheffield Wednesday', 'Southampton', 'Tottenham Hotspur', 
        'Wimbledon'
    ]
    
    # Dropdowns for selecting home and away teams
    home_team = st.selectbox('Please select Home Team:', clubs)
    away_team = st.selectbox('Please select the Away Team:', [team for team in clubs if team != home_team])
    
    # Add some spacing between the dropdowns and the chart
    st.write('\n')  # Adding a newline for spacing
    
    # Call the utility function to get probabilities
    home_win_prob, draw_prob, away_win_prob = calculate_win_lose_draw(home_team, away_team)
    
    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [home_win_prob, draw_prob, away_win_prob], color=['green', 'yellow', 'red'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f'{home_team} Winning', 'Draw', f'{away_team} Winning'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Game Outcome Probabilities', fontweight='bold')
    
    st.pyplot(fig)
