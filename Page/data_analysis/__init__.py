from .public_trans import static, pflow

import streamlit as st


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
        st.sidebar.subheader('Data Analysis Projects')
        page = st.sidebar.selectbox(
            'Please choose the target project!',
            self.pages,
            format_func=lambda page: page['title']
        )
        page['function']()


def app():
    sub_app = MultiPage()

    sub_app.add_page('Static Bus Line', static.app)
    sub_app.add_page('Bus Passenger Flow Analysis', pflow.app)

    sub_app.run()
