import streamlit as st
import pandas as pd
import os
import folium
import datetime
import altair as alt
from folium import plugins
import streamlit.components.v1

from conf.settings import DATA_PATH_PARKING

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def load_data():
    parking = pd.read_csv(os.path.join(DATA_PATH_PARKING, 'birmingham.csv'))
    loc = pd.read_csv(os.path.join(DATA_PATH_PARKING, 'bmh_location.csv'))

    # 删除没有位置信息的停车场时间序列数据
    parking = parking[parking['SystemCodeNumber'].isin(loc['SystemCodeNumber'])]

    # 删除时间序列过短的停车场数据，共2个
    temp_group = parking.groupby(by='SystemCodeNumber')
    system_code_number = list(set(parking['SystemCodeNumber']))
    for code, df in temp_group:
        if df.shape[0] < 1000:
            system_code_number.remove(code)
    parking = parking[parking['SystemCodeNumber'].isin(system_code_number)]
    loc = loc[loc['SystemCodeNumber'].isin(system_code_number)]
    loc.reset_index(drop=True, inplace=True)

    # 去除重复的时空数据，即一个停车场同一时间只能有一条数据
    parking.drop_duplicates(subset=['SystemCodeNumber', 'LastUpdated'], inplace=True)  # 根据给定字段去重
    parking.reset_index(drop=True, inplace=True)
    parking['LastUpdated'] = pd.to_datetime(parking['LastUpdated'], format="%Y/%m/%d %H:%M")
    parking.sort_values(by=['SystemCodeNumber', 'LastUpdated'], inplace=True)

    return parking, loc


def create_occupancy(data):
    parking = data.copy()

    # 剩余停车位 = 总停车位 - 占用停车位
    parking['Vacant'] = parking['Capacity'] - parking['Occupancy']

    # 停车占有率
    parking['OccupancyRate'] = parking['Occupancy'] / parking['Capacity']

    return parking


def create_occupancy_correlation(data, loc):
    parking = data[['LastUpdated', 'SystemCodeNumber', 'OccupancyRate']].copy()

    # 以 LastUpdated 为行索引，SystemCodeNumber 为列索引，OccupancyRate 为值，构建dataframe
    # 缺失值会以 NaN 填充
    parking_space = pd.pivot(parking, values='OccupancyRate', index='LastUpdated', columns='SystemCodeNumber')
    # 删除缺失值过多的行，保留的每行至少有15个非空值
    parking_space.dropna(axis=0, thresh=15, inplace=True)
    # 填充缺失值：用后非缺失值填充，再用前非缺失值填充
    parking_space.fillna(method='bfill', inplace=True)
    parking_space.fillna(method='ffill', inplace=True)
    # 计算相关性矩阵
    corr_matrix = parking_space.corr()
    # 将相关性矩阵连接到位置信息表
    locations_processed = pd.merge(left=loc, right=corr_matrix, left_on='SystemCodeNumber', right_index=True)

    # 保存数据
    locations_processed.to_csv(
        path_or_buf=os.path.join(DATA_PATH_PARKING, 'location_spacial_corr.csv'),
        encoding='utf-8')
    parking_space.to_csv(
        path_or_buf=os.path.join(DATA_PATH_PARKING, 'occupancy_time_series.csv'),
        encoding='utf-8')

    return parking_space, locations_processed


def plot_heat_map(time_series, loc_process):
    # 创建底图
    lon, lat = loc_process['longtitude'].mean(), loc_process['latitude'].mean()
    m = folium.Map(
        tiles='OpenStreetMap',
        location=(lat, lon),
        zoom_start=14
    )

    # 动态热力图数据
    system_code_number = loc_process['SystemCodeNumber'].to_list()
    time_list = []
    download_prog = st.progress(0)
    i = 0
    for time_step in range(time_series.shape[0]):
        parking_list = []
        for parking_id in system_code_number:
            parking_list.append([
                loc_process.loc[loc_process['SystemCodeNumber'] == parking_id, 'latitude'].values[0],  # 纬度
                loc_process.loc[loc_process['SystemCodeNumber'] == parking_id, 'longtitude'].values[0],  # 经度
                time_series[parking_id][time_step],  # 值
            ])
        time_list.append(parking_list)
        i += 1 / len(time_series)
        download_prog.progress(i)

    # 动态热力图时间戳
    data_space = time_series.reset_index()
    time_index = list(data_space['LastUpdated'].astype(dtype="str"))

    radius = st.slider('Please Choose HeatMap radius:', 30, 100, 60)
    # 动态热力图参数：
    # data 为三层嵌套列表，内部第一层列表数量对应时间戳 index 参数；内部第二层列表为 [lat,lon,value]
    # index 时间戳列表，时间为字符串格式
    folium.plugins.HeatMapWithTime(
        data=time_list,
        index=time_index,
        auto_play=True,
        radius=radius,
    ).add_to(m)

    # 创建folium图像，采用streamlit html解析组件解析
    fig_folium = folium.Figure().add_child(m)
    streamlit.components.v1.html(
        fig_folium.render(),  # 转化类型
        height=600,  # 设置显示高度，宽度自适应
    )


def plot_altair(data, loc):
    parking = data.copy()

    # Part1 --- 处理数据
    # 合并数据
    merge_data = pd.merge(parking, loc, on='SystemCodeNumber')
    merge_data['LastUpdated'] = pd.to_datetime(merge_data['LastUpdated'], format="%Y/%m/%d %H:%M")
    merge_data['weekday'] = merge_data['LastUpdated'].dt.weekday
    merge_data['hour'] = merge_data['LastUpdated'].dt.hour
    merge_data['IsWeekend'] = (merge_data['weekday'] == 5) | (merge_data['weekday'] == 6)  # 注意从0开始编号，bool值直接作为结果

    # 格式化数据，按 SystemCodeNumber,weekday,hour 分组，之后重新设置坐标，原三层坐标自动填充
    merge_data = merge_data.groupby(['SystemCodeNumber', 'weekday', 'hour']).mean()
    merge_data = merge_data.reset_index()

    merge_data['datetime'] = pd.to_datetime(
        (merge_data['weekday'] + 1).astype("str") + ' ' + (merge_data['hour']).astype("str"),
        format='%d %H',
    ) - datetime.timedelta(hours=8)  # 设置显示北京时间

    # Part 2 --- 可视化  ??
    # 定义选择器
    selection = alt.selection(fields=['SystemCodeNumber'], type='single', on='mouseover', nearest=True)
    # 定义颜色配置
    color_scale = alt.Scale(domain=[True, False], range=['#F5B041', '#5DADE2'])
    # 定义全局配置
    base = alt.Chart(merge_data).properties(
        width=350,
        height=200
    ).add_selection(selection)

    # 1. 位置散点图
    scatter = base.mark_circle().encode(
        x=alt.X(
            'mean(longtitude)',
            scale=alt.Scale(domain=(merge_data['longtitude'].min(), merge_data['longtitude'].max()))
        ),
        y=alt.Y(
            'mean(latitude)',
            scale=alt.Scale(domain=(merge_data['latitude'].min(), merge_data['latitude'].max()))
        ),
        color=alt.condition(
            selection,
            alt.value("lightgray"),
            "mean(OccupancyRate):Q",
            legend=None
        ),
        size=alt.Size('mean(OccupancyRate):Q', legend=None),
        tooltip=['SystemCodeNumber', 'mean(OccupancyRate):Q'],
    )

    # 2. 时间序列图
    sequential = base.mark_line().encode(
        x='datetime:T',
        y=alt.Y('mean(OccupancyRate):Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('weekday:N', legend=None),
    ).transform_filter(selection)

    # 3. 置信区间图
    line = base.mark_line().encode(
        x='hour',
        y=alt.Y('mean(OccupancyRate):Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('IsWeekend', legend=None, scale=color_scale)
    ).transform_filter(selection)

    band = base.mark_errorband(extent='ci').encode(
        x='hour',
        y=alt.Y('OccupancyRate:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('IsWeekend', legend=None, scale=color_scale)
    ).transform_filter(selection)

    # 4. 气泡表格图
    table = base.mark_circle().encode(
        x='hours(datetime):O',
        y='day(datetime):O',
        size=alt.Size('mean(OccupancyRate):Q', legend=None),
        color=alt.Color('IsWeekend', legend=None, scale=color_scale),
        tooltip='mean(OccupancyRate):Q',
    ).transform_filter(selection)

    fig = alt.vconcat(
        (scatter | sequential),
        ((line + band) | table),
    )

    return fig


def app():
    st.header('Time Series Analysis')

    # Part 1 --- 数据加载及预处理
    st.write('#### :key: Loading Data And Preprocessing')
    temp = st.info('Loading Parking Data And Location Data...')
    parking_data, location_data = load_data()
    temp.success('Loading Data Over!')

    col_left, col_right = st.columns((1, 1))
    with col_left.expander('Check the processed parking data:'):
        st.write(parking_data)
    with col_right.expander('Check the processed location data:'):
        st.write(location_data)

    # 创建占有率指标
    temp = st.info('Create Occupancy Indicator...')
    parking_data = create_occupancy(parking_data)
    # 计算占有率时间序列相关性
    temp.info('Create Spatial Correlation...')
    time_series, loc_processed = create_occupancy_correlation(parking_data, location_data)
    temp.success('Create Over!')

    col_left, col_center, col_right = st.columns((1, 1, 1))
    with col_left.expander('Check the occupancy indicator:'):
        st.write(parking_data)
    with col_center.expander('Check the occupancy time series:'):
        st.write(time_series)
    with col_right.expander('Check the location spacial correlation:'):
        st.write(loc_processed)

    # Part 2 --- 绘制停车占有率动态热力图
    st.write('---')
    st.write('#### :key: Occupancy Time Series HeatMap')

    temp = st.info('Plot Heat Map...')
    plot_heat_map(time_series, loc_processed)
    temp.success('Plot Over!')

    # Part 3 --- 时间序列空间自相关性分析
    st.write('---')
    st.write('#### :key: Spatial Auto-correlation Analysis Of Time Series')

    fig_altair = plot_altair(parking_data, loc_processed)
    st.altair_chart(fig_altair, use_container_width=True)  # fig_altair 不属于 altair.vegalite.v2.api.Chart 类型，因此没法自适应宽度

    pass


if __name__ == '__main__':
    app()
