import streamlit as st

# Page title with soccer ball icon
st.set_page_config(page_title='Soccer Predictor', page_icon='⚽')
st.title('⚽ Soccer Predictor')

# Sidebar menu
st.sidebar.title('Menu')

# Per Game section
st.sidebar.header('Per Game')
st.sidebar.subheader('[Use Case 1: Goals Scored](#use-case-1-goals-scored)')
st.sidebar.subheader('[Use Case 2: Win-Lose-Draw Probability](#use-case-2-win-lose-draw-probability)')

# Per League section
st.sidebar.header('Per League')
st.sidebar.subheader('[Use Case 3: League Standing](#use-case-3-league-standing)')
st.sidebar.subheader('[Use Case 4: League Predictor](#use-case-4-league-predictor)')
st.sidebar.subheader('[Use Case 5: Which Manager Is Better?](#use-case-5-which-manager-is-better)')

# Main content
def display_use_case(title, description):
    st.header(title)
    st.write(description)

# Placeholder text for the use cases
placeholder_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean consequat ex quis neque porttitor pharetra. Curabitur eu odio congue, pretium justo ut, consequat enim. Donec sollicitudin mauris nec diam consectetur rhoncus.
"""

# Use Case 1: Goals Scored
if st.sidebar.button('Use Case 1: Goals Scored'):
    display_use_case('Use Case 1: Goals Scored', placeholder_text)

# Use Case 2: Win-Lose-Draw Probability
if st.sidebar.button('Use Case 2: Win-Lose-Draw Probability'):
    display_use_case('Use Case 2: Win-Lose-Draw Probability', placeholder_text)

# Use Case 3: League Standing
if st.sidebar.button('Use Case 3: League Standing'):
    display_use_case('Use Case 3: League Standing', placeholder_text)

# Use Case 4: League Predictor
if st.sidebar.button('Use Case 4: League Predictor'):
    display_use_case('Use Case 4: League Predictor', placeholder_text)

# Use Case 5: Which Manager Is Better?
if st.sidebar.button('Use Case 5: Which Manager Is Better'):
    display_use_case('Use Case 5: Which Manager Is Better', placeholder_text)

