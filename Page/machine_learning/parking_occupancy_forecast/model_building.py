import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np

from conf.settings import DATA_PATH_PARKING


def pre_processing(data, loc, parking):
    sw_len = 18  # 8:00 - 16:30 的数据长度，滑动窗口长度
    batch_size = 32  # LSTM 模型批处理数量

    # Part 1 --- 加载数据，确定特征与标签
    # 特征值：所有停车场占有率序列 （，20）
    # 标签值：所选择停车场占有率时间序列 （，1）
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

    # Part 2 --- 划分训练集和测试集
    temp = st.info('Split training set and test set...')
    train_ratio = st.slider('Please choose train set ratio: ', 0, 100, 80)
    X_train, y_train, X_test, y_test = split_train_test(train_ratio / 100, X, y)

    # 划分结果可视化
    alt_data = data.reset_index()
    alt_data['index'] = alt_data.index
    alt_data['train_test'] = ['Train' if x <= len(y_train) else 'test' for x in alt_data.index]  # 添加训练集和测试集标签
    line = alt.Chart(alt_data).mark_line().encode(
        x='index:Q',
        y=f'{parking}:Q',
        color=alt.Color('train_test:N', legend=None),
    )
    st.altair_chart(line, use_container_width=True)
    temp.success('Split training set and test set over!')

    # Part 3 --- 构造时间序列数据集
    temp = st.info('Construct a time series dataset...')
    st.write(f'LSTM slide windows length: {sw_len}')
    train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=sw_len)
    test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=sw_len)
    with st.expander(
            f'Time series feature dimension: (train length, slide window length, feature dimension): {train_dataset.shape}'):
        st.write(train_dataset)
    with st.expander(f'Time series label dimension: (train length, label dimension): {train_labels.shape}'):
        st.write(train_labels.T)
    temp.success('Construct a time series dataset over!')

    pass


def create_dataset(x, y, seq_len=10):
    features = []
    labels = []
    for i in range(0, len(x) - seq_len, 1):
        # 利用前 seq_len 长度时间步预测后一时间步的值
        # 序列数据：维度为 （seq_len,20）
        data = x.iloc[i:i + seq_len]
        # 标签数据：维度为 （1，1）
        label = y.iloc[i + seq_len]
        features.append(data)
        labels.append(label)

    return np.array(features), np.array(labels)


def split_train_test(ratio, x_data, y_data):
    x_len = len(x_data)  # 特征数据集X的样本数量
    train_data_len = int(x_len * ratio)  # 训练集的样本数量
    x_train = x_data[:train_data_len]  # 训练集
    y_train = y_data[:train_data_len]  # 训练标签集
    x_test = x_data[train_data_len:]  # 测试集
    y_test = y_data[train_data_len:]  # 测试集标签集

    return x_train, y_train, x_test, y_test


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
