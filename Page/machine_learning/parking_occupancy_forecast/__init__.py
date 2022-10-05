import streamlit as st

from . import data_geographic, data_parking, model_building


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


def parking_of_app():
    sub_app = MultiPage()

    sub_app.add_page('Spatial Feature Analysis', data_geographic.app)
    sub_app.add_page('Time Series Analysis', data_parking.app)
    sub_app.add_page('LSTM Prediction', model_building.app)

    sub_app.run()
