import streamlit as st
import numpy as np
from PIL import Image
from leagues import get_league_metadata
import os

def show():
    st.header('Use Case 3: League Standing')
    st.write('What is the probability of a given team to end up in n-th position?')

    # Get league metadata
    clubs, num_league_positions = get_league_metadata('epl')

    # Dropdown for selecting a team
    team = st.selectbox('Select a team:', ['Select Team'] + clubs)

    # Dropdown for selecting league position
    position = st.selectbox('Select a league position:', ['Select Position'] + list(range(1, 21)))

    # Display the probability and the appropriate image
    if team == 'Select Team' or position == 'Select Position':
        probability = 'NA'
        image = Image.open('media/football_field.jpg')
    else:
        probability = f"{np.random.rand():.3f}"
        image_path = f"media/logos/EPL/{team}.png"
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = Image.open('media/football_field.jpg')  # Fallback image if logo not found

    # Display the probability
    st.markdown(f"""
        <div style='display: flex; align-items: center;'>
            <div>
                <p style='font-size: 24px; font-weight: bold; margin-bottom: 0;'>Probability</p>
                <p style='font-size: 20px; color: black;'>{probability}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display the image
    st.image(image, caption='', use_column_width=True)

