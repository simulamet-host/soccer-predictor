import streamlit as st
import importlib

# Page title with soccer ball icon
st.set_page_config(page_title='Soccer Predictor', page_icon='⚽')

# Initialize session state to store the selected use case
if 'selected_case' not in st.session_state:
    st.session_state.selected_case = None

# Sidebar menu
st.sidebar.title('Menu')

# Home button to return to main menu
if st.sidebar.button('Home'):
    st.session_state.selected_case = None

# Per Game section
st.sidebar.header('Per Game')
if st.sidebar.button('Use Case 1: Goals Scored'):
    st.session_state.selected_case = 'UseCase1_GoalsScored'
if st.sidebar.button('Use Case 2: Win-Lose-Draw Probability'):
    st.session_state.selected_case = 'win_lose_draw_probability'

# Per League section
st.sidebar.header('Per League')
if st.sidebar.button('Use Case 3: League Standing'):
    st.session_state.selected_case = 'league_standing'
if st.sidebar.button('Use Case 4: League Predictor'):
    st.session_state.selected_case = 'league_predictor'
if st.sidebar.button('Use Case 5: Which Manager Is Better?'):
    st.session_state.selected_case = 'which_manager_is_better'

# Display the selected use case content
if st.session_state.selected_case:
    case_module = importlib.import_module(f'use_cases.{st.session_state.selected_case}')
    case_module.show()
else:
    st.title('⚽ Soccer Predictor')
    st.write("Welcome to the Soccer Predictor dashboard! Please select a use case from the sidebar.")
