import streamlit as st

# Page title with soccer ball icon
st.set_page_config(page_title='Soccer Predictor', page_icon='⚽')

# Define a dictionary to store the titles and descriptions for each use case
use_cases = {
    'Home': ('Welcome to the Soccer Predictor dashboard! Please select a use case from the sidebar.', None),
    'Use Case 1: Goals Scored': ('Goals Scored Description...', 'pages.UseCase1_GoalsScored'),
    'Use Case 2: Win-Lose-Draw Probability': ('Win-Lose-Draw Probability Description...', 'pages.UseCase2_Win_Lose_Draw_Probability'),
    'Use Case 3: League Standing': ('League Standing Description...', 'pages.UseCase3_LeagueStanding'),
    'Use Case 4: League Predictor': ('League Predictor Description...', 'pages.UseCase4_LeaguePredictor'),
    'Use Case 5: Which Manager Is Better?': ('Which Manager Is Better Description...', 'pages.UseCase5_WhichManagerIsBetter')
}

# Sidebar menu
st.sidebar.title('Menu')

# Home button to return to main menu
if st.sidebar.button('Home'):
    st.session_state.selected_case = 'Home'

# Per Game section
st.sidebar.header('Per Game')
if st.sidebar.button('Use Case 1: Goals Scored'):
    st.session_state.selected_case = 'Use Case 1: Goals Scored'
if st.sidebar.button('Use Case 2: Win-Lose-Draw Probability'):
    st.session_state.selected_case = 'Use Case 2: Win-Lose-Draw Probability'

# Per League section
st.sidebar.header('Per League')
if st.sidebar.button('Use Case 3: League Standing'):
    st.session_state.selected_case = 'Use Case 3: League Standing'
if st.sidebar.button('Use Case 4: League Predictor'):
    st.session_state.selected_case = 'Use Case 4: League Predictor'
if st.sidebar.button('Use Case 5: Which Manager Is Better?'):
    st.session_state.selected_case = 'Use Case 5: Which Manager Is Better'

# Display the selected use case content
selected_case = st.session_state.get('selected_case', 'Home')
title, module_path = use_cases[selected_case]

if module_path:
    # Dynamically import and execute the module's `app` function
    module = __import__(module_path, fromlist=[''])
    module.app()
else:
    st.title('⚽ Soccer Predictor')
    st.write(title)
