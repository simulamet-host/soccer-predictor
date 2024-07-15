import streamlit as st
import numpy as np
from PIL import Image

def show():
    st.header('League Table Probability')
    st.write('What is the probability of a given team to end up in n-th position?')

    # List of football clubs
    clubs = [
        'Arsenal', 'Aston Villa', 'Blackburn Rovers', 'Chelsea', 'Coventry City', 
        'Crystal Palace', 'Everton', 'Ipswich Town', 'Leeds United', 'Liverpool', 
        'Manchester City', 'Manchester United', 'Middlesbrough', 'Norwich City', 
        'Nottingham Forest', 'Oldham Athletic', 'Queens Park Rangers', 
        'Sheffield United', 'Sheffield Wednesday', 'Southampton', 'Tottenham Hotspur', 
        'Wimbledon'
    ]
    
    # Dropdown for selecting a team
    team = st.selectbox('Select a team:', clubs)
    
    # Dropdown for selecting league position
    position = st.selectbox('Select a league position:', list(range(1, 21)))
    
    # Simulated probability calculation
    probability = np.random.rand()
    
    # Display the probability
    st.markdown(f"""
        <div style='display: flex; align-items: center;'>
            <div>
                <p style='font-size: 24px; font-weight: bold; margin-bottom: 0;'>Probability</p>
                <p style='font-size: 20px; color: black;'>{probability:.3f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Adding the image to the right side
    image = Image.open('football_field.jpg')
    st.image(image, caption='', use_column_width=True)

