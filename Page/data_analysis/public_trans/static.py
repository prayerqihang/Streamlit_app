import streamlit as st
from streamlit_folium import folium_static
from plotly import graph_objs as go
import os
import json
import folium

from conf.settings import DATA_PATH_STATIC
from lib import converter


def create_map(stops, polyline):
    # 创建底图
    aver_lat = [float(dic['location'].strip().split(',')[1]) for dic in stops]
    aver_lon = [float(dic['location'].strip().split(',')[0]) for dic in stops]
    center_lon, center_lat = converter.gcj02_to_wgs84(
        sum(aver_lon) / len(aver_lon),
        sum(aver_lat) / len(aver_lat)
    )

    my_map = folium.Map(
        location=(center_lat, center_lon),
        zoom_start=13.5,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # 添加站点
    for dic in stops:
        lon, lat = converter.gcj02_to_wgs84(
            float(dic['location'].strip().split(',')[0]),
            float(dic['location'].strip().split(',')[1]),
        )
        folium.Marker(
            location=[lat, lon],
            popup=f"{dic['name']}站",
            icon=folium.Icon(
                color='blue', icon='glyphicon-map-marker', icon_color='black'
            )
        ).add_to(my_map)

    # 添加线路
    loc_list = [
        converter.gcj02_to_wgs84(float(loc.split(',')[0]), float(loc.split(',')[1]))[::-1]
        for loc in polyline.split(';')
    ]

    folium.PolyLine(
        locations=loc_list,
        popup='Bus Line',
        color='darkblue',
        weight=5,
        opacity=0.5,
    ).add_to(my_map)

    return my_map


def plot_stop_line(stops):
    name_list = [dic['name'] for dic in stops]
    y_left = [2 * i for i in range(0, len(name_list), 2)]
    y_right = [2 * i for i in range(1, len(name_list), 2)]
    layout = go.Layout(
        autosize=False,
        width=500,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig = go.Figure(layout=layout)
    fig.add_traces([
        go.Scatter(
            x=[2 for _ in range(len(y_left))], y=y_left,
            text=name_list[0::2], textposition="bottom right", mode="lines+text"),
        go.Scatter(
            x=[2 for _ in range(len(y_right))], y=y_right,
            text=name_list[1::2], textposition="top left", mode="lines+text"),
    ])
    fig.update_traces(showlegend=False)
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig)


def app():
    st.header('Static Bus Line :oncoming_bus:')

    # Part 1 --- 选择需要展示的公交线路，显示 json 源代码信息
    st.write('#### :key: Choose Target Bus Line')

    file_name_list = os.listdir(DATA_PATH_STATIC)
    bus_line_name = [n.replace('.json', '') for n in file_name_list]

    bus_line_selectbox = st.selectbox(
        'Following bus_line are more in Huqiu District, Suzhou',
        bus_line_name,
        help='Specific data from GaoDe map'
    )

    bus_line_file = os.path.join(DATA_PATH_STATIC, bus_line_selectbox) + '.json'
    with open(bus_line_file, mode='rt', encoding='utf-8') as f:
        json_code = f.read()
        info_dic = json.loads(json_code)

    with st.expander('You can check bus_line information in detail here', False):
        st.json(json_code)

    # Part 2 --- 基本信息展示
    st.write('---')
    st.write('#### :key: Basic Information')

    bus_line_str = info_dic['polyline']
    bus_stops_list = info_dic['busstops']

    c_left, c_right = st.columns((1, 1))
    with c_left:
        st.write(f"**Bus Name**------{info_dic['name'].split('(')[0]}")
        st.write(f"**Start Stop**------{info_dic['start_stop']}")
        st.write(f"**Basic Price**-----{info_dic['basic_price']} yuan")

    with c_right:
        st.write(f"**Distance**-------{info_dic['distance']} km")
        st.write(f"**End Stop**-------{info_dic['end_stop']}")
        st.write(f"**Total Price**-----{info_dic['total_price']} yuan")

    _, c_center, _ = st.columns((0.2, 1, 0.2))
    with c_center:
        plot_stop_line(bus_stops_list)

    # Part 3 --- 地图可视化
    st.write('---')
    st.write('#### :key: Bus Line&Stops Visualization')
    st.write(
        '**The following shows the positions of bus line and bus stops, using the coordinate system as WGS84.**')
    c_l, c, c_r = st.columns((0.1, 0.8, 0.1))
    with c:
        m = create_map(bus_stops_list, bus_line_str)
        folium_static(m)


if __name__ == '__main__':
    app()
