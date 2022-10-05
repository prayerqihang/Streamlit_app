import streamlit as st
from .public_trans import public_trans_app
from .medical_accessibility import medical_acc_app


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
        st.sidebar.subheader('Data Analysis Project')
        page = st.sidebar.selectbox(
            'Go to:',
            self.pages,
            format_func=lambda page: page['title']
        )

        page['function']()

        # 显示各项目补充信息


def app():
    sub_app = MultiPage()

    sub_app.add_page('Medical Accessibility Analysis', medical_acc_app)
    sub_app.add_page('Public Trans', public_trans_app)

    sub_app.run()
