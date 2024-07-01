import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Simulated probabilities for testing purposes
    home_win_prob = np.random.uniform(0.3, 0.6)
    draw_prob = np.random.uniform(0.2, 0.4)
    away_win_prob = 1 - home_win_prob - draw_prob
    
    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [home_win_prob, draw_prob, away_win_prob], color=['green', 'yellow', 'red'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f'{home_team} Winning', 'Draw', f'{away_team} Winning'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Game Outcome Probabilities', fontweight='bold')
    
    st.pyplot(fig)
