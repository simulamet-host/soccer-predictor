import streamlit as st
# Create this file in the root directory to handle the multi-page navigation.

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
        app = st.sidebar.radio(
            'Go to',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()
