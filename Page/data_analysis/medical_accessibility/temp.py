import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import leafmap.foliumap as lfp
import folium
import osmnx as ox
from shapely import geometry
import networkx as nx
import branca

from conf.settings import DATA_PATH_MEDICAL


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data():
    # Part 1 --- 数据加载
    gdf_dict = {}  # 数据存储字典

    temp = st.info('Loading Geographic Data...')
    gdf_medical = gpd.read_file(
        filename=os.path.join(DATA_PATH_MEDICAL, '急救中心', '急救中心.shp'),
        encoding='utf-8'
    )
    gdf_edges = gpd.read_file(
        filename=os.path.join(DATA_PATH_MEDICAL, 'roads', 'Edges.shp'),
        encoding='gbk',  # shp文件采用gbk格式编码
    )
    gdf_nodes = gpd.read_file(
        filename=os.path.join(DATA_PATH_MEDICAL, 'roads', 'Nodes.shp'),
        encoding='gbk',  # shp文件采用gbk格式编码
    )
    gdf_zones = gpd.read_file(
        filename=os.path.join(DATA_PATH_MEDICAL, 'roads', 'Zones.shp'),
        encoding='utf-8',  # shp文件采用utf-8格式编码
    )
    temp.success('Loading Data Over!')

    # Part 2 --- 确定研究范围
    temp = st.info('Selecting Data In Target Zone...')
    west, south, east, north = lfp.gdf_bounds(gdf_medical)  # 获取gdf数据的边界经纬度
    west_east = (east - west) / 4
    south_north = (north - south) / 4
    west, south, east, north = west - west_east, south - south_north, east + west_east, north + south_north
    poly_df = gpd.GeoDataFrame({
        'value': [1],
        'geometry': geometry.box(minx=west, maxx=east, miny=south, maxy=north),
    }).set_crs(gdf_medical.crs)

    # Part 3 --- 筛选研究范围中的数据
    overlay_nodes = gpd.overlay(df1=gdf_nodes, df2=poly_df, how='intersection', keep_geom_type=True)
    overlay_edges = gpd.overlay(df1=gdf_edges, df2=poly_df, how='intersection', keep_geom_type=True)
    overlay_zones = gpd.overlay(df1=gdf_zones, df2=poly_df, how='intersection', keep_geom_type=True)

    # 删除 MultiLineString 对象
    overlay_edges.drop(
        labels=overlay_edges[overlay_edges.geom_type == 'MultiLineString'].index.to_list(),
        axis=0,
        inplace=True
    )
    overlay_edges.reset_index(drop=True, inplace=True)

    # 删除没有坐标信息的 LineString 对象
    overlay_edges.drop(
        labels=[i for i, g in enumerate(overlay_edges.geometry) if len(g.coords) == 0],
        axis=0,
        inplace=True
    )
    overlay_edges.reset_index(drop=True, inplace=True)
    temp.success('Select Over!')

    # Part 4 --- 创建要素属性
    temp = st.info('Creating Element Features...')
    gdf_dict.update({
        'medical': gdf_medical, 'nodes': overlay_nodes, 'edges': overlay_edges, 'zones': overlay_zones
    })
    gdf_attr_dict = create_attr(gdf_dict)

    temp.success('Create Over!')

    return gdf_attr_dict


def map_plot(gdf_dict):
    # Part 1 --- 用户选择地图属性
    temp = st.info('Customize Visual Map')

    col_left, col_right = st.columns((2, 3))
    with col_left:
        base_map = st.radio(
            label='Please choose the BaseMap:',
            options=['HYBRID', 'SATELLITE', 'OpenStreetMap'],
            index=0,
        )
    with col_right:
        chosen_layer = st.multiselect(
            label='Please choose the target layers:',
            options=gdf_dict.keys(),
            default=['medical']
        )
    temp.success(f'You choose the {base_map} map. You add the layers {chosen_layer[-1]}')

    # Part 2 --- 绘制
    temp = st.info('Plot...')
    m = lfp.Map()
    m.add_basemap(basemap=base_map)
    m.zoom_to_gdf(gdf=gdf_dict['medical'])

    for layer in chosen_layer:
        # 以下具体参数参见源代码
        if layer == 'medical':
            m.add_gdf(
                gdf=gdf_dict['medical'],
                layer_name='First Aid Center',
                zoom_to_layer=False,
                marker=folium.Marker(
                    icon=folium.Icon(color='green', icon_color='red', icon='glyphicon-plus')
                ),  # 可以传入类型为：Marker, Circle, CircleMarker
            )
        if layer == 'nodes':
            colormap_node = branca.colormap.LinearColormap(
                vmax=gdf_dict['nodes']['delay'].max(),
                vmin=gdf_dict['nodes']['delay'].min(),
                colors=['green', 'yellow', 'red'],
                caption='Intersection Delay'
            )  # 根据节点 delay 创建颜色条
            m.add_gdf(
                gdf=gdf_dict['nodes'],
                layer_name='Intersection',
                zoom_to_layer=False,
                marker=folium.CircleMarker(radius=1),
                style_function=lambda x: {'color': colormap_node(x["properties"]["delay"])},
            )
        if layer == 'edges':
            colormap_node = branca.colormap.LinearColormap(
                vmax=gdf_dict['edges']['travel_time'].max(),
                vmin=gdf_dict['edges']['travel_time'].min(),
                colors=['green', 'yellow', 'red'],
                caption='Travel Time'
            )
            m.add_gdf(
                gdf=gdf_dict['edges'],
                layer_name='Road Segment',
                zoom_to_layer=False,
                style_function=lambda x: {'color': colormap_node(x["properties"]["travel_time"])},
            )
        if layer == 'zones:':
            m.add_gdf(
                gdf=gdf_dict['zones'],
                layer_name='Zones',
                zoom_to_layer=False,
            )

    folium.LayerControl(collapsed=False).add_to(m)  # 多图层控制开关
    m.to_streamlit()

    temp.success('Plot Over!')

    pass


def create_attr(gdf_dict):
    # Part1 1 --- 添加节点属性：类型、延误
    nodes = gdf_dict['nodes']
    nodes['delay'] = 0
    for i in range(nodes.shape[0]):
        node_type = nodes['类型'].iloc[i]
        if node_type == '信号控制交叉口':
            nodes['delay'][i] = 40
        elif node_type == '主路优先权交叉口':
            nodes['delay'][i] = 20
        elif node_type == '无控制交叉口':
            nodes['delay'][i] = 15

    # Part 2 --- 添加边属性：行程时间
    edges = gdf_dict['edges']
    edges['travel_time'] = edges['长度'] * 3.6 / 60

    # Part 3 --- 添加区域属性：人口密度

    return gdf_dict


def network_analysis(gdf_dict):
    overlay_nodes, overlay_edges = gdf_dict['nodes'], gdf_dict['edges']

    # nodes信息为点名称列表，edges信息为起点(u)、终点(v)、权重(weight)三列的数据表形式
    nodes = list(overlay_nodes['编号'].drop_duplicates())  # 去重
    edges = overlay_edges[['起点', '终点', '长度']]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(ebunch_to_add=edges.values)

    # 添加节点属性：类型、延误
    for i in graph.nodes:
        node_type = overlay_nodes[overlay_nodes['编号'] == i]['类型'].to_list()
        if len(node_type) == 0:
            node_type.append('未知信息节点')
        if node_type[0] == '信号控制交叉口':
            delay = 40
        elif node_type[0] == '主路优先权交叉口' or node_type[0] == '无控制交叉口':
            delay = 25
        else:
            delay = 0
        graph.add_node(i, attr={'类型': node_type[0], '延误': delay})

    # 添加边属性
    edge_attr_df = pd.DataFrame(overlay_edges.copy())
    edge_attr_df.drop(labels='geometry', axis=1, inplace=True)
    edge_attr_df.set_index(keys=['起点', '终点', '长度'], drop=True, inplace=True)

    for multi_index in edge_attr_df.index.to_list():
        u, v, length = multi_index
        attr_dict = edge_attr_df.loc[(u, v, length), :].to_dict()
        for key, value in attr_dict.items():
            graph[u][v][key] = value

    return graph


def app():
    st.header('Visualization')

    # Part 1 --- 加载数据，预处理
    st.write('#### :key: Load Geographic Data And Pre-processing')
    data_dict = load_data()

    # Part 2 --- 可视化
    st.write('---')
    st.write('#### :key: Geographic Information Visualization')
    map_plot(data_dict)

    # Part 3 --- 将数据转化为 graph 对象


if __name__ == '__main__':
    app()
