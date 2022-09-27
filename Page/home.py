import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


# 加载页面动画
@st.cache
def load_lottie(url):
    res = requests.get(url)
    if res.status_code != 200:
        return None
    return res.json()


# 页面渲染CSS
@st.cache(suppress_st_warning=True)
def load_css(css_file):
    with open(css_file, mode='rt', encoding='utf_8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def app():
    load_css('style/home.css')  # 加载css
    lottie_coding = load_lottie('https://assets2.lottiefiles.com/packages/lf20_tqjrovxh.json')  # 加载动画
    img_test = Image.open('images/test.png')  # 加载图片

    # Part 1 --- 简单介绍
    with st.container():
        st.subheader('Hello, I am prayer-qihang :tiger:')
        st.write('### A student from SEU major in traffic engineering')
        st.write(
            '''
            I am passionate about the application of Python language in various fields, 
            including data analysis, machine learning, software development and so on.
            The site is a personal testing site that will include personal projects on machine learning and data analysis.
            '''
        )

    # Part 2 --- 网站内容简介
    with st.container():
        st.write('---')
        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader('What you can see in the website')
            st.write(
                '''
                #### :bulb: Data Analysis Projects:
                + Actual traffic issues
                + Data acquisition method based on crawler
                + Visualization of spatial and temporal variation characteristics of traffic parameters
                #### :bulb: Machine Learning Projects:
                + Actual traffic issues
                + Prediction methods based on various machine learning algorithms
                '''
            )
        with right_column:
            st_lottie(lottie_coding, height=380, key='coding')

    # Part 3 --- 主要项目介绍
    st.write('---')
    st.subheader('Projects Overview')

    with st.container():
        st.write('#### :key: Pubic Transportation Analysis')
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(img_test)
        with text_column:
            st.write(
                '''
                content
                '''
            )

    with st.container():
        st.write('#### :key: MNL')
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(img_test)
        with text_column:
            st.write(
                '''
                content
                '''
            )

    # Part 4 --- 联系方式
    st.write('---')
    st.subheader('Contact Me')

    form = st.form(key="annotation")

    with form:
        cols = st.columns((1, 1))
        name = cols[0].text_input('Your name:')
        profession = cols[1].text_input('Your profession:', help='Not necessary')
        st.text_input('Your Phone/Email/WeChat/QQ:')
        comment = st.text_area(
            "If you have any Additional information, please write down below:",
            help='Not necessary'
        )
        submitted = st.form_submit_button(label="Submit")

    pass
