import streamlit as st
from multiapp import MultiApp
from pages import UseCase1_GoalsScored, UseCase2_Win_Lose_Draw_Probability, UseCase3_LeagueStanding, UseCase4_LeaguePredictor, UseCase5_WhichManagerIsBetter

# Page title with soccer ball icon
st.set_page_config(page_title='Soccer Predictor', page_icon='⚽')

app = MultiApp()

# Add all your applications (use cases) here
app.add_app("Home", lambda: st.write("Welcome to the Soccer Predictor dashboard! Please select a use case from the sidebar."))
app.add_app("Use Case 1: Goals Scored", UseCase1_GoalsScored.app)
app.add_app("Use Case 2: Win-Lose-Draw Probability", UseCase2_Win_Lose_Draw_Probability.app)
app.add_app("Use Case 3: League Standing", UseCase3_LeagueStanding.app)
app.add_app("Use Case 4: League Predictor", UseCase4_LeaguePredictor.app)
app.add_app("Use Case 5: Which Manager Is Better?", UseCase5_WhichManagerIsBetter.app)

# Run the main app
app.run()
