import streamlit as st
from . import test


class MultiPage:
    def __init__(self):
        self.pages = []
        pass

    def add_page(self, title, func):
        self.pages.append(
            {
                'title': title,
                'function': func,
            }
        )

    def run(self):
        page = st.sidebar.radio(
            'Go to: Test',
            self.pages,
            format_func=lambda page: page['title']
        )

        page['function']()


def test_app():
    sub_app = MultiPage()

    sub_app.add_page('Test App', test.app)

    sub_app.run()
