import streamlit as st

from . import temp


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
            'Please Choose The Part You Want:',
            self.pages,
            format_func=lambda page: page['title']
        )

        page['function']()


def medical_acc_app():
    sub_app = MultiPage()

    sub_app.add_page('temp', temp.app)

    sub_app.run()
