import streamlit as st

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
if st.session_state.selected_case:
    st.experimental_set_query_params(page=st.session_state.selected_case)
    st.experimental_rerun()
else:
    st.title('⚽ Soccer Predictor')
    st.write("Welcome to the Soccer Predictor dashboard! Please select a use case from the sidebar.")
