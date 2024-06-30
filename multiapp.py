import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.sidebar.title('Menu')

        st.sidebar.header('Per Game')
        if st.sidebar.button('Use Case 1: Goals Scored'):
            self.apps[1]['function']()
        elif st.sidebar.button('Use Case 2: Win-Lose-Draw Probability'):
            self.apps[2]['function']()

        st.sidebar.header('Per League')
        if st.sidebar.button('Use Case 3: League Standing'):
            self.apps[3]['function']()
        elif st.sidebar.button('Use Case 4: League Predictor'):
            self.apps[4]['function']()
        elif st.sidebar.button('Use Case 5: Which Manager Is Better?'):
            self.apps[5]['function']()
        else:
            self.apps[0]['function']()  # Default to Home if no button is pressed
