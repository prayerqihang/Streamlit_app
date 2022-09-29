import streamlit as st
from .parking_occupancy_forecast import parking_of_app


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
        st.sidebar.subheader('Machine Learning Project')
        page = st.sidebar.selectbox(
            'Go to:',
            self.pages,
            format_func=lambda page: page['title']
        )

        page['function']()

        # 显示各项目补充信息
        if page['title'] == 'Parking Occupancy Forecast':
            st.sidebar.title('About')
            st.sidebar.info(
                """
                This project is mainly made by the seuteer. If you want
                to know more, please access [GitHub](https://github.com/seuteer)
                """
            )


def app():
    sub_app = MultiPage()

    sub_app.add_page('Parking Occupancy Forecast', parking_of_app)

    sub_app.run()
