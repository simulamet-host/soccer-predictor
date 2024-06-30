import streamlit as st

# Page title with soccer ball icon
st.set_page_config(page_title='Soccer Predictor', page_icon='⚽')

# Initialize session state to store the selected use case
if 'selected_case' not in st.session_state:
    st.session_state.selected_case = None

# Define a dictionary to store the titles and descriptions for each use case
use_cases = {
    'Use Case 1: Goals Scored': 'Goals Scored Description...',
    'Use Case 2: Win-Lose-Draw Probability': 'Win-Lose-Draw Probability Description...',
    'Use Case 3: League Standing': 'League Standing Description...',
    'Use Case 4: League Predictor': 'League Predictor Description...',
    'Use Case 5: Which Manager Is Better?': 'Which Manager Is Better? Description...'
}

# Sidebar menu
st.sidebar.title('Menu')

# Home button to return to main menu
if st.sidebar.button('Home'):
    st.session_state.selected_case = None

# Per Game section
st.sidebar.header('Per Game')
for case in ['Use Case 1: Goals Scored', 'Use Case 2: Win-Lose-Draw Probability']:
    if st.sidebar.button(case):
        st.session_state.selected_case = case

# Per League section
st.sidebar.header('Per League')
for case in ['Use Case 3: League Standing', 'Use Case 4: League Predictor', 'Use Case 5: Which Manager Is Better?']:
    if st.sidebar.button(case):
        st.session_state.selected_case = case

# Display the selected use case content
if st.session_state.selected_case:
    st.header(st.session_state.selected_case)
    st.write(use_cases[st.session_state.selected_case])
else:
    st.title('⚽ Soccer Predictor')
    st.write("Welcome to the Soccer Predictor dashboard! Please select a use case from the sidebar.")
