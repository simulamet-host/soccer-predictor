import streamlit as st
import pandas as pd
from utils.UseCase4_utils import generate_probability_matrix
from leagues import get_league_metadata

def show():
    st.header('Use Case 4: League Predictor')
    st.write('This table shows the probabilities of each EPL team finishing in each league position for the current season. The values in the table are probabilities that add up to 1 in both rows and columns.')
    
    # Get league metadata
    clubs, num_league_positions = get_league_metadata('epl')
    
    # Generate the probability matrix
    probability_matrix = generate_probability_matrix(size=20)
    
    # Create a DataFrame for better display in Streamlit
    df = pd.DataFrame(probability_matrix, index=clubs, columns=[f"**{i}**" for i in range(1, 21)])
    
    # Apply styling to the DataFrame
    df.index = [f"**{club}**" for club in clubs]
    
    # Display the table with Streamlit
    st.markdown(df.to_markdown(), unsafe_allow_html=True)

# Call the function to display the UI
show()
