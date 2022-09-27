import streamlit as st
import datetime
from Page import home, machine_learning, data_analysis


# 创建类实现分页功能
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
        st.sidebar.title('App Navigation')
        # 添加单选框
        page = st.sidebar.radio(
            'Please choose the target page you want!',
            self.pages,
            format_func=lambda page: page['title']
        )
        st.sidebar.write('---')

        # 只在主页显示联系方式
        if page['title'] == 'Home':
            st.sidebar.title("About")
            st.sidebar.info(
                """
                You can follow me on social media: [GitHub]() | [Blog](). 
                Or contact me directly through the following way: 
                Email - 213193391@seu.edu.cn | Phone - 13116637275
                """
            )

            st.sidebar.info(f'''
            Current time {st.session_state.date_time.date()} / {st.session_state.date_time.time()}
            ''')

        # 运行页面主函数
        page['function']()


# set_page_config函数只能被调用一次，且必须为第一个调用的函数
st.set_page_config(page_title='TransPyNavigator', page_icon=':car:', layout='wide')
st.title('Transportation & Python Navigator')


# 会话状态：其中定义的信息会在页面刷新时重置，因此可以利用它，可以非常巧妙地配置每次访问页面时的全局信息
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
else:
    st.session_state.first_visit = False

# 初始化全局配置
# st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(
#     hours=8)  # Streamlit_Cloud的时区是UTC，加8小时即北京时间
st.session_state.date_time = datetime.datetime.now().replace(microsecond=0)  # 仅用于本地运行
if st.session_state.first_visit:
    st.balloons()  # 一次访问时才会放气球

# 实例化
app = MultiPage()

# 添加app
app.add_page('Home', home.app)
app.add_page('Machine Learning', machine_learning.app)
app.add_page('Data Analysis', data_analysis.app)

if __name__ == '__main__':
    app.run()
