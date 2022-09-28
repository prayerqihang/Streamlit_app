import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np
import tensorflow as tf
import folium
from folium import plugins
from tensorflow import keras
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import streamlit.components.v1 as components

from conf.settings import DATA_PATH_PARKING


def model_predict(parking, train_data, test_data, train_label, test_label):
    model_path = os.path.join(DATA_PATH_PARKING, 'models', parking)
    if not os.path.exists(model_path):
        st.error('Model is not exists, please train a model first!')
    else:
        model = tf.keras.models.load_model(model_path)

        col_left, col_right = st.columns((1, 1))
        with col_left.expander('Train Dataset Prediction:', expanded=True):
            # 训练集的预测
            train_pred = model.predict(train_data)
            plot_predict(train_label, train_pred)
        with col_right.expander('Test Dataset Prediction:', expanded=True):
            # 测试集的预测
            test_pred = model.predict(test_data)
            plot_predict(test_label, test_pred)

    return train_pred, test_pred


def model_train(parking, train_dataset, train_batch_dataset, test_batch_dataset, epochs=30):
    # ??
    # 如果模型存在，则直接加载
    model_path = os.path.join(DATA_PATH_PARKING, 'models', parking)
    if os.path.exists(model_path):
        temp = st.info('Loading model from cloud...')
        model = tf.keras.models.load_model(model_path)
        temp.success('Model and weights have been loaded successfully!')
    # 如果模型不存在，则训练模型
    else:
        temp = st.info('Training LSTM model...')
        model = keras.Sequential([
            keras.layers.LSTM(128, input_shape=train_dataset.shape[-2:], return_sequences=True),
            keras.layers.Dropout(0.5),
            keras.layers.LSTM(64),
            keras.layers.Dense(1)  # 全连接层，输出为1
        ])

        model.compile(optimizer='adam', loss="mse")
        res = model.fit(
            train_batch_dataset,
            epochs=epochs,
            validation_data=test_batch_dataset,
            # callbacks=[tensorboard_callback],
            verbose=0,
        )  # 沉默输出
        model.save(model_path, save_format='h5')
        temp.success('Model train over!')


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
    train_ratio = st.slider('Please choose train set ratio(%): ', 0, 100, 80)
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
    train_dataset, train_labels, train_index = create_dataset(X_train, y_train, seq_len=sw_len)
    test_dataset, test_labels, test_index = create_dataset(X_test, y_test, seq_len=sw_len)

    with st.expander(
            f'Time series feature dimension: (train length, slide window length, feature dimension): {train_dataset.shape}'):
        st.write(train_dataset)
    with st.expander(f'Time series label dimension: (train length, label dimension): {train_labels.shape}'):
        st.write(train_labels.T)
    temp.success('Construct a time series dataset over!')

    # Part 4 --- 构造批处理数据集 # ??
    temp = st.info('Construct batch dataset...')
    st.write(f'LSTM batch size: {batch_size}')
    train_batch_dataset = create_batch_data(train_dataset, train_labels, batch_size=batch_size)
    test_batch_dataset = create_batch_data(test_dataset, test_labels, train=False, batch_size=batch_size)
    temp.success('Construct batch dataset over!')

    return train_dataset, train_labels, train_batch_dataset, train_index, test_dataset, test_labels, test_batch_dataset, test_index


def create_batch_data(x, y, train=True, buffer_size=100, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))  # 数据封装，tensor类型
    if train:  # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:  # 测试集
        return batch_data.batch(batch_size)


def create_dataset(x, y, seq_len=10):
    features = []
    labels = []
    labels_index = []  # 用于记录测试集每一数据的时间戳，方便绘制热力图
    for i in range(0, len(x) - seq_len, 1):
        # 利用前 seq_len 长度时间步预测后一时间步的值
        # 序列数据：维度为 （seq_len,20）
        data = x.iloc[i:i + seq_len]
        # 标签数据：维度为 （1，1）
        label = y.iloc[i + seq_len]
        features.append(data)
        labels.append(label)
        labels_index.append(label.name)  # 字符串列表

    return np.array(features), np.array(labels), labels_index


def split_train_test(ratio, x_data, y_data):
    x_len = len(x_data)  # 特征数据集X的样本数量
    train_data_len = int(x_len * ratio)  # 训练集的样本数量
    x_train = x_data[:train_data_len]  # 训练集
    y_train = y_data[:train_data_len]  # 训练标签集
    x_test = x_data[train_data_len:]  # 测试集
    y_test = y_data[train_data_len:]  # 测试集标签集

    return x_train, y_train, x_test, y_test


def plot_predict(label, pred):
    r2 = r2_score(label, pred)  # 计算拟合优度
    rmse = np.sqrt(MSE(label, pred))  # 计算均方误差
    st.write(f'R2: {round(r2, 3)}', f'RMSE: {round(rmse, 3)}')
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(len(label)), label, c='r', alpha=0.8, label='True Value')
    plt.plot(range(len(pred)), pred, c='b', label='Predicted Value')
    plt.legend(fontsize=10)
    st.pyplot(fig)


def plot_heat_map(location, parking, data, time_index, radius):
    # 创建底图
    parking_lat = location.loc[location['SystemCodeNumber'] == parking, 'latitude'].values[0]
    parking_lon = location.loc[location['SystemCodeNumber'] == parking, 'longtitude'].values[0]
    m = folium.Map(
        tiles='OpenStreetMap',
        location=(parking_lat, parking_lon),
        zoom_start=14
    )

    # 动态热力图数据
    time_list = []
    download_prog = st.progress(0)
    i = 0
    for time_step in range(len(time_index)):
        parking_list = []
        parking_list.append([
            parking_lat,
            parking_lon,
            float(data[time_step]),  # 值
        ])
        time_list.append(parking_list)
        i += 1 / len(time_index)
        download_prog.progress(round(i, 1))

    folium.plugins.HeatMapWithTime(
        data=time_list[::3],
        index=time_index[::3],
        auto_play=True,
        radius=radius,
    ).add_to(m)

    # 创建folium图像，采用streamlit html解析组件解析
    fig_folium = folium.Figure().add_child(m)
    components.html(
        fig_folium.render(),  # 转化类型
        height=300,  # 设置显示高度，宽度自适应
    )


def app():
    st.header('Parking Occupancy Forecast——LSTM Prediction')

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

    train_dataset, train_labels, train_batch_dataset, train_label_index, \
    test_dataset, test_labels, test_batch_dataset, test_label_index = \
        pre_processing(time_series, location, target_parking)

    # Part 3 --- 模型训练
    st.write('---')
    st.write('#### :key: Model Training')

    model_train(target_parking, train_dataset, train_batch_dataset, test_batch_dataset, epochs=30)

    # Part 4 --- 模型预测
    st.write('---')
    st.write('#### :key: Model Predicting')

    train_prediction, test_prediction = model_predict(target_parking, train_dataset, test_dataset, train_labels,
                                                      test_labels)

    # Part 5 --- 预测结果可视化
    st.write('---')
    st.write('#### :key: Visualization Of Prediction Results')

    temp = st.info('Ploting Heat Map...')

    radius = st.slider('Please Choose HeatMap radius:', 60, 140, 100)

    col_left, _, col_right = st.columns((0.49, 0.02, 0.49))
    with col_left.expander('Origin Test Set Labels:', expanded=True):
        plot_heat_map(location, target_parking, test_labels, test_label_index, radius)
    with col_right.expander('Predict Test Set Values:', expanded=True):
        plot_heat_map(location, target_parking, test_prediction, test_label_index, radius)

    temp.success('Plot Successfully!')


if __name__ == '__main__':
    app()
