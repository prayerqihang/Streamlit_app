import streamlit as st
import os

from conf.settings import DATA_PATH_PASSENGER


def app():
    st.header('Bus Passenger Flow Analysis ')

    # Part 1 --- 选择线路
    st.write('#### :key: Choose Target Bus Line')

    file_name_list = os.listdir(DATA_PATH_PASSENGER)
    bus_line_name = [n.replace('.csv', '') for n in file_name_list if n.startswith('line')]

    bus_line_selectbox = st.selectbox(
        'Following bus_line are more in Huqiu District, Suzhou',
        bus_line_name,
        help='The data is not public and is for data analysis only'
    )
    pass
