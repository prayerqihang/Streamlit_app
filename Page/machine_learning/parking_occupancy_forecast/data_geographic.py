import streamlit as st
import os
import geopandas as gpd
import osmnx as ox
import pandas as pd
import leafmap.foliumap as lfp
import networkx as nx
from osmnx import plot, speed
from PIL import Image
import branca
import folium

from conf.settings import DATA_PATH_PARKING, IMAGE_PATH


# streamlit 缓存装饰器，在首次运行函数后会对：函数的输入，函数中的外部参数，函数体，函数体中调用的函数
# 进行缓存，再次运行时可以跳过该函数的运行过程，直接从缓存中取数据
# 参数：suppress_st_warning False-当函数体中存在st命令时，会给出警告；True-忽略警告
# 参数：allow_output_mutation True-再次运行时，允许缓存中的数据发生变化
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def download_osm_data(gpkg_path):
    dict_layer = {}  # 存储地理数据的字典，按图层存储

    if not os.path.exists(gpkg_path):
        # 如果本地不存在数据，则从osm进行下载

        # Part 1 --- 停车场点要素
        temp = st.info('Parking Download...')
        df = gpd.read_file(os.path.join(DATA_PATH_PARKING, 'bmh_location.csv'))
        df[['longtitude', 'latitude']] = df[['longtitude', 'latitude']].apply(pd.to_numeric)  # 将对象转化为数字类型
        dict_layer['parking'] = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df['longtitude'], df['latitude']),
            crs='EPSG:4326'  # 设置坐标系 WGS84
        )
        temp.success('Parking Data Download Over!')

        # 根据停车场经纬度确定并扩大研究范围
        west, south, east, north = lfp.gdf_bounds(dict_layer['parking'])
        dew, dns = (east - west) / 2, (north - south) / 2
        west, south, east, north = west - dew, south - dns, east + dew, north + dns

        # Part 2 --- 道路网络
        temp = st.info('Roads Download...')
        graph = ox.graph_from_bbox(north, south, east, west)  # 根据坐标范围获取道路网
        # osmnx 目前仅支持输出文件类型（存在问题，暂不清楚原因）： ESRI Shapfile, osm, graphml, GPKG
        ox.save_graphml(graph, filepath=os.path.join(DATA_PATH_PARKING, 'bmh.graphml'))
        dict_layer['nodes'], dict_layer['edges'] = ox.graph_to_gdfs(graph)
        temp.success('Roads Data Download Over!')

        # Part 3 --- POI
        temp = st.info('POIs Download...')
        pois = ox.geometries_from_bbox(north, south, east, west, tags={'amenity': True})
        dict_layer['pois'] = pois[pois['geometry'].type.isin(['Point'])]  # 只选择其中的点要素
        temp.success('POIs Data Download Over!')

        # Part 4 --- 建筑区域
        temp = st.info('Buildings Download...')
        buildings = ox.geometries_from_bbox(north, south, east, west, tags={'building': True})
        dict_layer['buildings'] = buildings[buildings['geometry'].type.isin(['Polygon'])]  # 只选择其中的面要素
        temp.success('Buildings Data Download Over!')
    else:
        # 如果本地存在数据，则直接读取
        for layer in ['parking', 'pois', 'buildings', 'nodes', 'edges']:
            dict_layer[layer] = gpd.read_file(gpkg_path, layer=layer)

    return dict_layer


def save_osm_data(dict_layer, gpkg_path):
    temp = st.info('Saving Geographic Data...')
    if not os.path.exists(gpkg_path):
        save_progress = st.progress(0)  # 保存进度
        p = 0
        for layer, gdf in dict_layer.items():
            temp.info(f'Saving layer: {layer}...')
            # geopandas导出数据存在数据格式问题，暂不清楚原因
            gdf = gdf.apply(lambda c: c.astype(str) if c.name != 'geometry' else c, axis=0)
            gdf.to_file(filename=gpkg_path, driver='GPKG', layer=layer)
            p += 1 / len(dict_layer)
            save_progress.progress(p)  # 进度由 0-1


def network_analysis(gpkg_path):
    # 如果不存在道路网文件，则抛出异常
    if not os.path.exists(gpkg_path.replace('.gpkg', '.graphml')):
        st.error('No road network file found, please check the data!')
    else:
        # 进行网络分析
        graph = ox.load_graphml(filepath=gpkg_path.replace('.gpkg', '.graphml'))

        # Part 1 --- 网络基本描述
        temp = st.info('Computing basic descriptive and topological metrics of the network...')
        G_proj = ox.project_graph(graph)  # 网络投影
        nodes_proj = ox.graph_to_gdfs(G_proj)  # 节点投影

        # nodes_proj为二元元组，第一个元素包含所有节点，第二个元素包含所有边
        graph_area_m = nodes_proj[0].unary_union.convex_hull.area  # ?
        stats_dict = ox.basic_stats(G_proj, area=graph_area_m)

        with st.expander('You can check information here', False):
            st.json(stats_dict)
        temp.success('Basic descriptive and topological metrics of the network are calculated!')

        col1, col2 = st.columns((1, 1))

        # Part 2 --- 计算道路长度，可视化
        with col1.expander('Road Length', expanded=True):
            filepath = os.path.join(IMAGE_PATH, 'fig_length.png')
            if not os.path.exists(filepath):
                # 若图片不存在，则重新绘制并保存
                ec_len = ox.plot.get_edge_colors_by_attr(graph, 'length', cmap='hot')
                fig_length, ax = ox.plot_graph(
                    graph, edge_color=list(ec_len), edge_linewidth=1, node_size=0, bgcolor='white'
                )
                fig_length.savefig(filepath, bbox_inches='tight')

                st.pyplot(fig=fig_length)
            else:
                # 存在，则直接读取
                st.image(Image.open(filepath))

        # Part 3 --- 计算道路中心度，可视化
        # closeness_centrality衡量节点之间的接近程度，中心度越高，距离越短
        with col2.expander('Road Centrality', expanded=True):
            filepath = os.path.join(IMAGE_PATH, 'fig_centrality.png')
            if not os.path.exists(filepath):
                edge_centrality = nx.closeness_centrality(nx.line_graph(graph))
                nx.set_edge_attributes(graph, edge_centrality, name='edge_centrality')
                ec_cen = ox.plot.get_edge_colors_by_attr(graph, 'edge_centrality', cmap='cool')
                fig_centrality, ax = ox.plot_graph(
                    graph, edge_color=list(ec_cen), edge_linewidth=1, node_size=0, bgcolor='white'
                )
                fig_centrality.savefig(filepath, bbox_inches='tight')

                st.pyplot(fig=fig_centrality)
            else:
                st.image(Image.open(filepath))

        # Part 4 --- 计算道路行驶速度，可视化
        col1, col2 = st.columns((1, 1))

        with col1.expander('Speed', expanded=True):
            filepath = os.path.join(IMAGE_PATH, 'fig_speed.png')
            if not os.path.exists(filepath):
                graph = ox.speed.add_edge_speeds(graph)  # 添加边属性 speed_kph，默认为每条边的平均最大自由流速度
                ec_speed = ox.plot.get_edge_colors_by_attr(graph, 'speed_kph', cmap='viridis')
                fig_speed, ax = ox.plot_graph(
                    graph, node_size=0, edge_color=list(ec_speed), edge_linewidth=1, bgcolor='white'
                )
                fig_speed.savefig(filepath, bbox_inches='tight')

                st.pyplot(fig=fig_speed)
            else:
                st.image(Image.open(filepath))

        # Part 5 --- 计算道路行驶时间，可视化
        with col2.expander('Travel Time', expanded=True):
            filepath = os.path.join(IMAGE_PATH, 'fig_time.png')
            if not os.path.exists(filepath):
                graph = ox.speed.add_edge_travel_times(graph)  # 添加边属性 travel_time，由speed_kph和length得到
                ec_time = ox.plot.get_edge_colors_by_attr(graph, 'travel_time', cmap='bone')
                fig_time, ax = ox.plot_graph(
                    graph, node_size=0, edge_color=list(ec_time), edge_linewidth=1, bgcolor='white'
                )
                fig_time.savefig(filepath, bbox_inches='tight')

                st.pyplot(fig=fig_time)
            else:
                st.image(Image.open(filepath))


def plot_leafmap(dict_layer):
    # Part 1 --- 用户选择
    temp = st.info('Customize Visual Map')

    col_left, col_right = st.columns((1, 2))
    # 选择底图
    with col_left:
        base_map = st.radio(
            'Please Choose BaseMap',
            ('OpenStreetMap', 'HYBRID'),
            index=0,  # 默认为 OpenStreetMap
        )
    # 选择需要查看图层
    with col_right:
        layer_chosen = st.multiselect(
            'Please Choose Layer (You are allowed to choose more than one)',
            list(dict_layer.keys()),
            default=['parking', 'nodes', 'edges', 'pois']  # 默认显示的图层
        )
    temp.success(f'You have chosen layer: {layer_chosen[-1]}')

    # Part 2 --- 地图绘制
    m = lfp.Map()
    m.add_basemap(basemap=base_map)
    m.zoom_to_gdf(dict_layer['nodes'])  # 地图缩放大小
    for layer in layer_chosen:
        if layer == 'parking':
            m.add_gdf(
                dict_layer['parking'],
                # tooltip=folium.GeoJsonTooltip(
                #     fields=['SystemCodeNumber'],
                #     aliases=['Parking Code'],  # fields的别名
                # ),  # 鼠标悬浮显示信息（存在问题，暂不清楚原因）
                marker=folium.Marker(
                    icon=folium.Icon(color='green', icon='car')
                ),
                layer_name='parking',
                zoom_to_layer=False,
            )  # 继承 folium.GeoJSON 类的参数
        if layer == 'nodes':
            m.add_gdf(
                dict_layer['nodes'],
                # popup=folium.GeoJsonPopup(
                #     fields=['street_count'],
                #     aliases=['Number of street connections']
                # ),  # 鼠标点击显示信息（存在问题，暂不清楚原因）
                marker=folium.CircleMarker(radius=1.5),
                layer_name='nodes',
                zoom_to_layer=False,
            )
        if layer == 'edges':
            m.add_gdf(
                dict_layer['edges'],
                layer_name='edges',
                zoom_to_layer=False
            )
        if layer == 'pois':
            # 绘制 PIO 热力图
            df_pois = lfp.gdf_to_df(dict_layer['pois']).copy()
            df_pois['longitude'] = dict_layer['pois']['geometry'].x  # 热力图经度
            df_pois['latitude'] = dict_layer['pois']['geometry'].y  # 热力图维度
            df_pois['value'] = 1  # 热力图值
            radius = st.slider('Please Choose HeatMap radius:', 5, 30, 15)
            m.add_heatmap(df_pois, value="value", radius=radius, name='pois', show=False)
        if layer == 'buildings':
            dict_layer['buildings'] = dict_layer['buildings'].to_crs(3875)  # 计算面积时，转化为投影坐标系
            dict_layer['buildings']['area'] = round(dict_layer['buildings'].area, 2)  # ?
            m.add_gdf(
                dict_layer['buildings'][['name', 'area', 'amenity', 'geometry']],
                # tooltip=folium.GeoJsonTooltip(
                #     fields=['name', 'area', 'amenity'],
                #     aliases=['Building name', 'Building Area(m^2)', 'Amenity']
                # ),
                layer_name='buildings',
                zoom_to_layer=False,
            )

    folium.LayerControl(collapsed=False).add_to(m)  # 参数：？
    m.to_streamlit()


def app():
    st.header('Spatial Feature Analysis')

    # Part 1 --- 地理数据获取
    st.write('#### :key: Geographic Data Acquisition')
    gpkg_path = os.path.join(DATA_PATH_PARKING, 'bmh.gpkg')  # 数据存储路径

    temp = st.info('Loading Cloud Data...')
    dict_layer_gdf = download_osm_data(gpkg_path)
    temp.success('Loading Cloud Data over!')

    # Part 2 ---保存数据
    st.write('---')
    st.write('#### :key: Save Data')
    save_osm_data(dict_layer_gdf, gpkg_path)
    temp.info('Saving Geographic Data Over!')

    # Part 3 --- 交通网络分析
    st.write('---')
    st.write('#### :key: Network Analysis')

    temp = st.info('Computing road length, centrality, travel time and speed...')
    network_analysis(gpkg_path)
    temp.success('Render the network according to road length, centrality, travel time and speed!')

    # Part 4 --- 地理数据可视化
    st.write('---')
    st.write('#### :key: Map Data Visualization')

    plot_leafmap(dict_layer_gdf)


if __name__ == '__main__':
    app()
