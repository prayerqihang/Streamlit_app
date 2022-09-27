import streamlit as st
import pandas as pd
import os

from conf.settings import DATA_PATH_PARKING


def pre_processing(data, loc, parking):
    train_ratio = 0.8  # 训练集占比
    sw_len = 18  # 8:00 - 16:30 的数据长度，滑动窗口长度
    batch_size = 32  #

    temp = st.info('Partition feature values and labels...')
    X = data
    y = data.loc[:, data.columns == parking]
    spatial_weight = loc[parking]

    col_left, col_right = st.columns((2, 1.5))
    with col_left.expander(f'Feature Dimension (length of time series, parking number): {X.shape}'):
        st.write(X)
    with col_right.expander(f'Label Dimension (length of time series, 1): {y.shape}'):
        st.write(y)
    st.write(f'{parking} Parking Spacial Weight:', pd.DataFrame(spatial_weight).T)
    temp.success('Partition feature values and labels over!')

    pass


def app():
    st.header('LSTM Prediction')

    # Part 1 --- 加载数据，选择停车场
    st.write('#### :key: Load Data')

    time_series = pd.read_csv(
        os.path.join(DATA_PATH_PARKING, 'occupancy_time_series.csv'),
        index_col='LastUpdated',  # 设置时间戳为行索引
    )
    location = pd.read_csv(os.path.join(DATA_PATH_PARKING, 'location_spacial_corr.csv'))

    target_parking = st.selectbox(
        'Please Choose The Target Parking:',
        time_series.columns
    )

    # Part 2 --- 数据处理
    st.write('---')
    st.write('#### :key: Processing Data')

    pre_processing(time_series, location, target_parking)

    pass


if __name__ == '__main__':
    app()
