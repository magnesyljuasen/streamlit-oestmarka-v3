import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
from folium.plugins import Fullscreen
import time
import base64

def streamlit_settings(title, icon):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(
        """<style>[data-testid="collapsedControl"] svg {height: 3rem;width: 3rem;}</style>""",
        unsafe_allow_html=True,
    )
   
    with open("src/styles/main.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def find_scenario_names(folder_path):
    scenario_name_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith("unfiltered.csv"):
            scenario_name = str(filename.split(sep="_")[0])
            scenario_name_list.append(scenario_name)
    return scenario_name_list

@st.cache_resource(show_spinner=False)
def read_position(filepath):
    df_position = pd.read_csv(filepath_or_buffer=f"{filepath}_unfiltered.csv", usecols=["x", "y", "har_adresse", "objectid", "profet_bygningstype", "bruksareal_totalt", "solceller", "grunnvarme", "fjernvarme", "luft_luft_varmepumpe", "oppgraderes", "bygningsomraadeid"])
    return df_position

@st.cache_resource(show_spinner=False)
def create_map(df_position):
    def add_wms_layer_to_map(url, layer, layer_name, opacity = 0.5, show = False):
        folium.WmsTileLayer(
            url = url,
            layers = layer,
            transparent = True, 
            control = True,
            fmt="image/png",
            name = layer_name,
            overlay = True,
            show = show,
            opacity = opacity
            ).add_to(folium_map)
        
    def add_marker_cluster_to_map():
        marker_cluster = MarkerCluster(
            name="1000 clustered icons",
            overlay=False,
            control=False,
            options={"disableClusteringAtZoom": 13},
        ).add_to(folium_map)
        return marker_cluster
    
    def styling_function(row):
        popup_text = ""
        text = ""
        rows = row[
            [
                "grunnvarme",
                "fjernvarme",
                "solceller",
                "luft_luft_varmepumpe",
                "oppgraderes",
            ]
        ]
        text = "".join(rows.index[rows].str[0].str.upper())
        TEXT_COLOR_MAP = {
            "G": "#48a23f",
            "F": "#1d3c34",
            "S": "#FFC358",
            "L": "black",
            "O": "#b7dc8f",
        }
        try:
            text_color = TEXT_COLOR_MAP[text]
        except Exception:
            text_color = "black"

        icon = folium.plugins.BeautifyIcon(
            # icon = "home",
            border_width=1,
            border_color=text_color,
            text_color=text_color,
            # background_color = "#FFFFFF00",
            icon_shape="circle",
            number=text,
            icon_size=(20, 20),
        )
        tooltip_text = f"""
            {row["har_adresse"]} (<strong>{row["bruksareal_totalt"]:,} m²</strong>)<br>
            <em>{row["profet_bygningstype"]}</em>""".replace(
            ",", " "
        )
        return popup_text, tooltip_text, icon
    
    def add_building_to_marker_cluster(marker_cluster, scenario_name, df):
        for index, row in df.iterrows():
            popup_text, tooltip_text, icon = styling_function(row)
            folium.Marker(
                [row["y"], row["x"]],
                # popup = popup_text,
                tooltip=tooltip_text,
                icon=icon,
            ).add_to(marker_cluster)
    
    def add_geojson_to_map(
        file_path,
        fill_color="#ff0000",
        color="#000000",
        weight=2,
        opacity=0.5,
        tooltip=None,
        layer_name=None,
        show=False,
        ):
        def style_function(feature):
            return {
                "fillColor": fill_color,
                "color": color,
                "weight": weight,
                "fillOpacity": opacity,
            }
        folium.GeoJson(
            file_path,
            style_function=style_function,
            tooltip=folium.Tooltip(text=tooltip),
            name=layer_name,
            show=show,
        ).add_to(folium_map)

    center_y = df_position["x"].mean()
    center_x = df_position["y"].mean()
    folium_map = folium.Map(
        location=[center_x, center_y],
        zoom_start=15,
        scrollWheelZoom=True,
        tiles=None,
        max_zoom=22,
        control_scale=True,
        prefer_canvas=True
        )           
    folium.TileLayer("CartoDB positron", name="Bakgrunnskart").add_to(folium_map)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", name="Flyfoto", attr="Flyfoto").add_to(folium_map)
    geometry = [Point(xy) for xy in zip(df_position['x'], df_position['y'])]
    gdf_buildings = gpd.GeoDataFrame(df_position, geometry=geometry, crs='EPSG:4326')
    #gdf_buildings = gdf_buildings.drop(columns=['y', 'x'])
    #geojson_buildings = gdf_buildings.to_json()
    marker_cluster = add_marker_cluster_to_map()
    add_building_to_marker_cluster(marker_cluster=marker_cluster, scenario_name=selected_scenario_name, df=df_position)
    add_wms_layer_to_map(
        url = "https://geo.ngu.no/mapserver/LosmasserWMS2?request=GetCapabilities&service=WMS",
        layer = "Losmasse_flate",
        layer_name = "Løsmasser",
        opacity = 0.5,
        show = False
        )
    add_wms_layer_to_map(
        url = "https://geo.ngu.no/mapserver/GranadaWMS5?request=GetCapabilities&service=WMS",
        layer = "Energibronn",
        layer_name = "GRANADA",
        opacity = 0.5,
        show = False
        )
    add_geojson_to_map(
            file_path="src/geojson/planområde.geojson",
            fill_color="transparent",
            color="black",
            weight=2,
            opacity=0.5,
            tooltip="Planområde",
            layer_name="Planområde",
            show=True,
        )
    add_geojson_to_map(
        file_path="src/geojson/utvidet_område.geojson",
        fill_color="transparent",
        color="black",
        weight=0.5,
        opacity=0.5,
        tooltip="Utvidet område",
        layer_name="Utvidet område",
        show=True,
    )
    add_geojson_to_map(
        file_path="src/geojson/eksisterende_bygg.geojson",
        fill_color="red",
        color="black",
        weight=1,
        opacity=0.5,
        tooltip="Eksisterende bygg",
        layer_name="Eksisterende bygg",
        show=False,
    )
    add_geojson_to_map(
        file_path="src/geojson/nye_bygg.geojson",
        fill_color="green",
        color="black",
        weight=1,
        opacity=0.5,
        tooltip="Nye bygg",
        layer_name="Nye bygg",
        show=False,
    )
    drawing = folium.plugins.Draw(
        position="topright",
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
        },
        edit_options={"featureGroup": None},
    )
    folium_map.add_child(drawing)
    drawing.add_to(folium_map)
    Fullscreen().add_to(folium_map)
    folium.LayerControl(position="bottomleft").add_to(folium_map)
    folium_map.options["attributionControl"] = False
    return folium_map, gdf_buildings

def display_map(folium_map):
    st_map = st_folium(
        folium_map,
        use_container_width=True,
        height=400,
        returned_objects=["last_active_drawing"],
        )
    return st_map

def spatial_join(gdf_buildings):
    try:
        polygon = Polygon(st_map["last_active_drawing"]["geometry"]["coordinates"][0])
        polygon_gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon])
        filtered_gdf = gpd.sjoin(gdf_buildings, polygon_gdf, op="within")
    except Exception:
        #st.info('Tegn et polygon for å gjøre et utvalg av bygg', icon="ℹ️")
        st.stop()
    return filtered_gdf

def scenario_comparison():
    with st.sidebar:
        scenario_comparison = st.checkbox("Sammenligne scenarier?", value=False)
    return scenario_comparison

def building_plan_filter(df_position):
    with st.sidebar:
        selected_buildings_option = st.radio(
            "Velg bygningsmasse", 
            options = [
                "Eksisterende bygningsmasse", 
                #"Planforslag (inkl. dagens bygg som skal bevares)", 
                #"Planforslag (ekskl. helsebygg)", 
                "Planforslag og områdene rundt Østmarka"
                ]
                )
        selected_buildings_option_map = {
            "Eksisterende bygningsmasse" : "E",
            "Planforslag (inkl. dagens bygg som skal bevares)" : "P1",
            "Planforslag (ekskl. helsebygg)" : "P2",
            "Planforslag og områdene rundt Østmarka" : "P3"
        }
        building_area_id = selected_buildings_option_map[selected_buildings_option]
    df_position = df_position[df_position['bygningsomraadeid'] == building_area_id]
    return df_position

@st.cache_resource(show_spinner=False)
def read_hourly_data(object_ids, filepath):    
    df_hourly_data = pd.read_csv(filepath_or_buffer=f"{filepath}_timedata.csv", usecols=object_ids)
    return df_hourly_data

def select_scenario():
    with st.sidebar:
        option_list = SCENARIO_NAMES.copy()
        option_list.remove('Referansesituasjon')
        scenario_name = st.radio(label='Velg scenario', options=option_list)
    return scenario_name

def get_dict_arrays(df_hourly_data):
    df_thermal = df_hourly_data[df_hourly_data['ID'] == '_termisk_energibehov'].reset_index(drop = True).drop('ID', axis=1).assign(sum=lambda x: x.sum(axis=1))
    df_electric = df_hourly_data[df_hourly_data['ID'] == '_elektrisk_energibehov'].reset_index(drop = True).drop('ID', axis=1).assign(sum=lambda x: x.sum(axis=1))
    df_spaceheating = df_hourly_data[df_hourly_data['ID'] == '_romoppvarming_energibehov'].reset_index(drop = True).drop('ID', axis=1).assign(sum=lambda x: x.sum(axis=1))
    df_dhw = df_hourly_data[df_hourly_data['ID'] == '_tappevann_energibehov'].reset_index(drop = True).drop('ID', axis=1).assign(sum=lambda x: x.sum(axis=1))
    df_elspecific = df_hourly_data[df_hourly_data['ID'] == '_elspesifikt_energibehov'].reset_index(drop = True).drop('ID', axis=1).assign(sum=lambda x: x.sum(axis=1))
    df_grid = df_hourly_data[df_hourly_data['ID'] == '_nettutveksling_energi_liste'].reset_index(drop = True).drop('ID', axis=1).assign(sum=lambda x: x.sum(axis=1))
    df_total = df_spaceheating + df_dhw + df_elspecific
    df_total_delivered = df_thermal + df_electric
    df_produced_heat = df_spaceheating + df_dhw - df_thermal
    df_produced_el = df_elspecific - df_electric
    df_thermal_total = df_spaceheating + df_dhw
    dict_arrays = {
        'thermal': df_thermal['sum'],
        'thermal_total' : df_thermal_total['sum'],
        'electric': df_electric['sum'],
        'spaceheating': df_spaceheating['sum'],
        'dhw': df_dhw['sum'],
        'elspecific': df_elspecific['sum'],
        'grid': df_grid['sum'],
        'total': df_total['sum'],
        'total_delivered' : df_total_delivered['sum'],
        'produced_heat': df_produced_heat['sum'],
        'produced_el' : df_produced_el['sum']}
    return dict_arrays

def hour_to_month(hourly_array, aggregation='sum'):
    result_array = []
    temp_value = 0 if aggregation in ['sum', 'max'] else []
    count = 0 if aggregation == 'average' else None
    for index, value in enumerate(hourly_array):
        if np.isnan(value):
            value = 0
        if aggregation == 'sum':
            temp_value += value
        elif aggregation == 'average':
            temp_value.append(value)
            count += 1
        elif aggregation == 'max' and value > temp_value:
            temp_value = value
        if index in [744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8759]:
            if aggregation == 'average':
                if count != 0:
                    result_array.append(sum(temp_value) / count)
                else:
                    result_array.append(0)
                temp_value = []
                count = 0
            else:
                result_array.append(temp_value)
                temp_value = 0 if aggregation in ['sum', 'max'] else []
    return result_array

def get_dict_months(dict_arrays, aggregation):
    dict_months = {
        'thermal': hour_to_month(dict_arrays['thermal'], aggregation),
        'thermal_total': hour_to_month(dict_arrays['thermal_total'], aggregation),
        'electric': hour_to_month(dict_arrays['electric'], aggregation),
        'spaceheating': hour_to_month(dict_arrays['spaceheating'], aggregation),
        'dhw': hour_to_month(dict_arrays['dhw'], aggregation),
        'elspecific': hour_to_month(dict_arrays['elspecific'], aggregation),
        'grid': hour_to_month(dict_arrays['grid'], aggregation),
        'total': hour_to_month(dict_arrays['total'], aggregation),
        'total_delivered': hour_to_month(dict_arrays['total_delivered'], aggregation),
        'produced_heat': hour_to_month(dict_arrays['produced_heat'], aggregation),
        'produced_el': hour_to_month(dict_arrays['produced_el'], aggregation)
    }
    return dict_months

def calculate_key_values(monthly_dict, aggregation='sum'):
    if aggregation == 'sum':
        value = sum(monthly_dict)
    elif aggregation == 'max':
        value = max(monthly_dict)
    return value

def get_key_values(monthly_dict, aggregation):
    dict_months = {
        'thermal' : calculate_key_values(monthly_dict['thermal'], aggregation),
        'thermal_total' : calculate_key_values(monthly_dict['thermal_total'], aggregation),
        'electric': calculate_key_values(monthly_dict['electric'], aggregation),
        'spaceheating': calculate_key_values(monthly_dict['spaceheating'], aggregation),
        'dhw': calculate_key_values(monthly_dict['dhw'], aggregation),
        'elspecific': calculate_key_values(monthly_dict['elspecific'], aggregation),
        'grid': calculate_key_values(monthly_dict['grid'], aggregation),
        'total': calculate_key_values(monthly_dict['total'], aggregation),
        'total_delivered': calculate_key_values(monthly_dict['total_delivered'], aggregation),
        'produced_heat': calculate_key_values(monthly_dict['produced_heat'], aggregation),
        'produced_el': calculate_key_values(monthly_dict['produced_el'], aggregation)
    }
    return dict_months

def metric(text, color, energy, effect, energy_reduction = 0, effect_reduction = 0):
    energy = int(round(energy, -3))
    effect = int(round(effect, 1))
    if energy_reduction > 0 or effect_reduction > 0:
        st.markdown(f"<span style='color:{color}'><small>{text}</small><br><big>**{energy:,}** kWh/år</big> ({-energy_reduction} %) | <big>**{effect:,}** kW</big> ({-effect_reduction} %)".replace(",", " "), unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:{color}'><small>{text}</small><br><big>**{energy:,}** kWh/år</big> | <big>**{effect:,}** kW</big>".replace(",", " "), unsafe_allow_html=True)

def download_link(df, filename='dataframe.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<small><a href="data:file/csv;base64,{b64}" download="{filename}">Last ned data</a>'
    return href

def energy_effect_plot():
    #metric(text = "Totalt", color = TOTAL_COLOR, energy = results[selected_scenario_name]["dict_sum"]["total"], effect = results[selected_scenario_name]["dict_max"]["total"])
    col1, col2, col3 = st.columns(3)
    with col1:
        metric(text = "Oppvarmingsbehov", color = SPACEHEATING_COLOR, energy = results[selected_scenario_name]["dict_sum"]["spaceheating"], effect = results[selected_scenario_name]["dict_max"]["spaceheating"])
    with col2:
        metric(text = "Tappevannsbehov", color = DHW_COLOR, energy = results[selected_scenario_name]["dict_sum"]["dhw"], effect = results[selected_scenario_name]["dict_max"]["dhw"])
    with col3:
        metric(text = "Elspesifikt behov", color = ELSPECIFIC_COLOR, energy = results[selected_scenario_name]["dict_sum"]["elspecific"], effect = results[selected_scenario_name]["dict_max"]["elspecific"])
    #with col4:
    #    metric(text = "Totalt behov", color = TOTAL_COLOR, energy = results[selected_scenario_name]["dict_sum"]["total"], effect = results[selected_scenario_name]["dict_max"]["total"])
    df = pd.DataFrame({
        "Måneder" : MONTHS,
        "Totalt (kW)" : results[selected_scenario_name]["dict_months_max"]["total"],
        "Oppvarming (kW)" : results[selected_scenario_name]["dict_months_max"]["spaceheating"],
        "Tappevann (kW)" : results[selected_scenario_name]["dict_months_max"]["dhw"],
        "Elspesifikt (kW)" : results[selected_scenario_name]["dict_months_max"]["elspecific"],
        "Totalt (kWh)" : results[selected_scenario_name]["dict_months_sum"]["total"],
        "Oppvarming (kWh)" : results[selected_scenario_name]["dict_months_sum"]["spaceheating"],
        "Tappevann (kWh)" : results[selected_scenario_name]["dict_months_sum"]["dhw"],
        "Elspesifikt (kWh)" : results[selected_scenario_name]["dict_months_sum"]["elspecific"],
    })
    y_max_energy = np.max(df["Totalt (kWh)"] * 1.1)
    y_max_effect = np.max(df["Totalt (kW)"] * 1.1)
    fig = go.Figure()
    COLORS = [DHW_COLOR, ELSPECIFIC_COLOR, SPACEHEATING_COLOR, TOTAL_COLOR]
    kWh_labels = ['Tappevann (kWh)', 'Elspesifikt (kWh)', 'Oppvarming (kWh)']
    kW_labels = ['Tappevann (kW)', 'Elspesifikt (kW)', 'Oppvarming (kW)', 'Totalt (kW)']
    for i, series in enumerate(kWh_labels):
        bar = go.Bar(x=df['Måneder'], y=df[series], name=series, yaxis='y', marker=dict(color=COLORS[i]))
        fig.add_trace(bar)
    for i, series in enumerate(kW_labels):
        fig.add_trace(go.Scatter(x=df['Måneder'], y=df[series], name=series, yaxis='y2', mode='lines+markers', line=dict(width=1, color="black", dash = "dot"), marker=dict(color=COLORS[i], symbol="diamond", line=dict(width=1, color = "black"))))
    fig.update_layout(
        showlegend=False,
        margin=dict(b=0, t=0),
        yaxis=dict(title=None, side='left', showgrid=True, tickformat=",.0f", range=[0, y_max_energy]),
        yaxis2=dict(title=None, side='right', overlaying='y', showgrid=True, range=[0, y_max_effect]),
        xaxis=dict(title=None, showgrid=True, tickformat=",.0f"),
        barmode='relative',
        yaxis_ticksuffix=" kWh",
        yaxis2_ticksuffix=" kW",
        separators="* .*",
        height=150
        )
    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': True, 'staticPlot': True})
    st.markdown(download_link(df = df, filename = "data.csv"), unsafe_allow_html=True)

def energy_effect_delivered_plot():
    st.caption("*Referansesituasjon*")
    HOURS = np.arange(0, 8760)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=HOURS,
            y=results[selected_scenario_name]["dict_arrays"]["total_delivered"],
            hoverinfo='skip',
            stackgroup="one",
            visible = 'legendonly',
            fill="tonexty",
            line=dict(width=0, color=TOTAL_COLOR),
            name=f'Fra strømnettet (totalt):<br>{int(round(results[selected_scenario_name]["dict_sum"]["total_delivered"],-3)):,} kWh/år<br>{int(round(results[selected_scenario_name]["dict_max"]["total_delivered"],-1)):,} kW'.replace(",", " ")
            ))
    fig.add_trace(
        go.Scatter(
            x=HOURS,
            y=results[selected_scenario_name]["dict_arrays"]["electric"],
            hoverinfo='skip',
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=ELECTRIC_COLOR),
            name=f'Elspesifikt (fra strømnettet):<br>{int(round(results[selected_scenario_name]["dict_sum"]["electric"],-3)):,} kWh/år<br>{int(round(results[selected_scenario_name]["dict_max"]["electric"],-1)):,} kW'.replace(",", " ")
            ))
    if results[selected_scenario_name]["dict_sum"]["produced_el"] > 1:
        fig.add_trace(
            go.Scatter(
                x=HOURS,
                y=results[selected_scenario_name]["dict_arrays"]["produced_el"],
                hoverinfo='skip',
                stackgroup="one",
                fill="tonexty",
                line=dict(width=0, color=PRODUCED_EL_COLOR),
                name=f'Strømproduksjon:<br>{int(round(results[selected_scenario_name]["dict_sum"]["produced_el"],-3)):,} kWh/år<br>{int(round(results[selected_scenario_name]["dict_max"]["produced_el"],-1)):,} kW'.replace(",", " ")
                ))
    fig.add_trace(
        go.Scatter(
            x=HOURS,
            y=results[selected_scenario_name]["dict_arrays"]["thermal"],
            hoverinfo='skip',
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=THERMAL_COLOR),
            name=f'Termisk (fra strømnettet):<br>{int(round(results[selected_scenario_name]["dict_sum"]["thermal"],-3)):,} kWh/år<br>{int(round(results[selected_scenario_name]["dict_max"]["thermal"],-1)):,} kW'.replace(",", " ")
            ))
    if results[selected_scenario_name]["dict_sum"]["produced_heat"] > 1:
        fig.add_trace(
            go.Scatter(
                x=HOURS,
                y=results[selected_scenario_name]["dict_arrays"]["produced_heat"],
                hoverinfo='skip',
                stackgroup="one",
                fill="tonexty",
                line=dict(width=0, color=PRODUCED_HEAT_COLOR),
                name=f'Varmeproduksjon:<br>{int(round(results[selected_scenario_name]["dict_sum"]["produced_heat"],-3)):,} kWh/år<br>{int(round(results[selected_scenario_name]["dict_max"]["produced_heat"],-1)):,} kW'.replace(",", " ")
                ))
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            orientation='h',
            xanchor='left',
            yanchor='top',
            title=None,
            bgcolor='rgba(255, 255, 255, 0.5)',
            font=dict(size=13)
        ),
        showlegend=True,
        margin=dict(b=0, t=0),
        barmode='relative',
        #yaxis_ticksuffix=" kW",
        separators="* .*",
        height=300,
        yaxis=dict(
            title="Effekt (kW)",
            side='left', 
            showgrid=True, 
            tickformat=",.0f", 
            range=[0, results[selected_scenario_name]["dict_max"]["total"] * 1.3]),
        xaxis = dict(
            tickmode = 'array', 
            tickvals = [0, 24 * (31), 24 * (31 + 28), 24 * (31 + 28 + 31), 24 * (31 + 28 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31)], 
            ticktext = ["1.jan", "", "1.mar", "", "1.mai", "", "1.jul", "", "1.sep", "", "1.nov", "", "1.jan"],
            title=None,
            showgrid=True)
        )
    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': True, 'staticPlot': True})
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label = "**Energi** fra strømnettet", value = f'{int(round(results[selected_scenario_name]["dict_sum"]["total_delivered"],-3)):,} kWh/år'.replace(",", " "), label_visibility='visible')
    with c2:
        st.metric(label = "**Effekt** fra strømnettet", value = f'{int(round(results[selected_scenario_name]["dict_max"]["total_delivered"],-1)):,} kW'.replace(",", " "), label_visibility='visible')
    
    with st.expander("Mer informasjon"):
        data = results[selected_scenario_name]["dict_months_sum"]
        df = pd.DataFrame(data)
        df["months"] = MONTHS
        st.write("Tabellen under viser månedlige verdier for energi (kWh).")
        st.dataframe(
            data=df,
            column_config={
                "months" : "Måned",
                "thermal" : st.column_config.NumberColumn("Romoppvarming + tappevann - varmeproduksjon (kWh)", format="%d"),
                "thermal_total" : st.column_config.NumberColumn("Romoppvarming + tappevann (kWh)", format="%d"),
                "spaceheating" : st.column_config.NumberColumn("Romoppvarming (kWh)", format="%d"),
                "elspecific" : st.column_config.NumberColumn("Elspesifikt (kWh)", format="%d"),
                "grid" : st.column_config.NumberColumn("Levert fra strømnettet (kWh)", format="%d"),
                "produced_heat" : st.column_config.NumberColumn("Levert varme (kWh)", format="%d"),
                "produced_el" : st.column_config.NumberColumn("Levert strøm (kWh)", format="%d"),
                "total" : st.column_config.NumberColumn("Totalt (kWh)", format="%d"),
                "total_delivered" : st.column_config.NumberColumn("Levert totalt (kWh)", format="%d"),
                "dhw" : st.column_config.NumberColumn("Tappevann (kWh)", format="%d"),
                "electric" : st.column_config.NumberColumn("Elspesifikt behov - lokalprodusert strøm (kWh)", format="%d")
            }, 
            use_container_width=True)
    
def energy_effect_scenario_plot():
    st.caption(f"*{selected_scenario_name}*")
    HOURS = np.arange(0, 8760)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=HOURS,
            y=results[selected_scenario_name]["dict_arrays"]["grid"],
            hoverinfo='skip',
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=ELECTRIC_COLOR),
            name=f'Fra strømnettet:<br>{int(round(results[selected_scenario_name]["dict_sum"]["grid"],-3)):,} kWh/år<br>{int(round(results[selected_scenario_name]["dict_max"]["grid"],-1)):,} kW'.replace(",", " ")
            ))
    renewable_array = np.array(results[selected_scenario_name]["dict_arrays"]["total_delivered"]) - np.array(results[selected_scenario_name]["dict_arrays"]["grid"])
    fig.add_trace(
        go.Scatter(
            x=HOURS,
            y=renewable_array,
            hoverinfo='skip',
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=RENEWABLE_COLOR),
            name=f'Fornybar energi:<br>{int(round(np.sum(renewable_array),-3)):,} kWh/år<br>{int(round(np.max(renewable_array),-1)):,} kW'.replace(",", " ")
            ))

    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            orientation='h',
            xanchor='left',
            yanchor='top',
            title=None,
            bgcolor='rgba(255, 255, 255, 0.5)',
            font=dict(size=13)
        ),
        showlegend=True,
        margin=dict(b=0, t=0),
        barmode='relative',
        #yaxis_ticksuffix=" kW",
        separators="* .*",
        height=300,
        yaxis=dict(
            title="Effekt (kW)",
            side='left', 
            showgrid=True, 
            tickformat=",.0f", 
            range=[0, results[selected_scenario_name]["dict_max"]["total"] * 1.5]),
        xaxis = dict(
            tickmode = 'array', 
            tickvals = [0, 24 * (31), 24 * (31 + 28), 24 * (31 + 28 + 31), 24 * (31 + 28 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30), 24 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31)], 
            ticktext = ["1.jan", "", "1.mar", "", "1.mai", "", "1.jul", "", "1.sep", "", "1.nov", "", "1.jan"],
            title=None,
            showgrid=True)
        )
    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': True, 'staticPlot': True})
    c1, c2 = st.columns(2)
    energy_reduction = int(round(((results[selected_scenario_name]["dict_sum"]["total_delivered"] - results[selected_scenario_name]["dict_sum"]["grid"])/results[selected_scenario_name]["dict_sum"]["total_delivered"])*100,1))
    effect_reduction = int(round(((results[selected_scenario_name]["dict_max"]["total_delivered"] - results[selected_scenario_name]["dict_max"]["grid"])/results[selected_scenario_name]["dict_max"]["total_delivered"])*100,1))
    with c1:
        st.metric(label = "**Energi** fra strømnettet", value = f'{int(round(results[selected_scenario_name]["dict_sum"]["grid"],-3)):,} kWh/år (-{energy_reduction:,}%)'.replace(",", " "), label_visibility='visible')
    with c2:
        st.metric(label = "**Effekt** fra strømnettet", value = f'{int(round(results[selected_scenario_name]["dict_max"]["grid"],-1)):,} kW (-{effect_reduction:,}%)'.replace(",", " "), label_visibility='visible')

    with st.expander("Mer informasjon"):
        st.write("...")
    
def energy_effect_comparison_plot():
    st.markdown(f"<span style='color:{AFTER_COLOR}'>Fremtidig behov fra strømnettet for alle scenariene (kWh/år og kW)".replace(",", " "), unsafe_allow_html=True)
    
#    col1, col2, col3 = st.columns(3)
#    with col1:
#        metric(text = f"Fremtidig behov fra strømnettet", color = AFTER_COLOR, energy = results[selected_scenario_name]["dict_sum"]["grid"], effect = results[selected_scenario_name]["dict_max"]["grid"])
#    with col2:
#        metric(text = "Dagens behov fra strømnettet", color = BEFORE_COLOR, energy = results[selected_scenario_name]["dict_sum"]["total_delivered"], effect = results[selected_scenario_name]["dict_max"]["total_delivered"])
#    with col3:
#        metric(text = "Historisk behov fra strømnettet", color = HISTORIC_COLOR, energy = results[selected_scenario_name]["dict_sum"]["total"], effect = results[selected_scenario_name]["dict_max"]["total"])
    sum_values = []
    max_values = []
    for scenario_name in SCENARIO_NAMES:
        sum_values.append(results[scenario_name]["dict_sum"]["grid"])
        max_values.append(results[scenario_name]["dict_max"]["grid"])
    
    df = pd.DataFrame({
        "scenario" : SCENARIO_NAMES,
        "effect" : max_values,
        "energy" : sum_values
    })
    df = df.sort_values(by='energy', ascending=False)
    reference_row = df[df['scenario'] == 'Referansesituasjon']
    df = df[df['scenario'] != 'Referansesituasjon']
    df = pd.concat([reference_row, df])
    df.reset_index(drop=True, inplace=True)
    y_max_energy = np.max(df["energy"] * 1.1)
    y_max_effect = np.max(df["effect"] * 1.1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["scenario"], y=df["energy"], marker=dict(color=AFTER_COLOR)))
    fig.add_trace(go.Scatter(x=df["scenario"], y=df["effect"], yaxis='y2', mode='lines+markers', line=dict(width=1, color="black", dash = "dot"), marker=dict(color=AFTER_COLOR, symbol="diamond", line=dict(width=1, color = "black"))))
    fig.update_layout(
        showlegend=False,
        margin=dict(b=0, t=0),
        yaxis=dict(title=None, side='left', showgrid=True, tickformat=",.0f", range=[0, y_max_energy]),
        yaxis2=dict(title=None, side='right', overlaying='y', showgrid=True, range=[0, y_max_effect]),
        xaxis=dict(title=None, showgrid=True, tickformat=",.0f"),
        yaxis_ticksuffix=" kWh/år",
        yaxis2_ticksuffix=" kW",
        separators="* .*",
        height=300
        )
    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': True, 'staticPlot': True})
    #st.markdown(download_link(df = df, filename = "data.csv"), unsafe_allow_html=True)

def duration_curve_plot():
    st.markdown(f"<span style='color:{AFTER_COLOR}'>Fremtidig behov fra strømnettet for alle scenariene som varighetskurver".replace(",", " "), unsafe_allow_html=True)
    keys = list(results.keys())
    data = []
    for key in keys:
        x_data = np.arange(8761)
        y_data = np.sort(results[key]['dict_arrays']['grid'])[::-1]
        trace = go.Scatter(x=x_data, y=y_data, mode='lines', name=key)
        data.append(trace)
    layout = go.Layout(
        margin=dict(b=0, t=0),
        height=300, 
        xaxis=dict(title=None, showgrid=True), 
        yaxis=dict(title=None, showgrid=True), 
        separators="* .*",
        showlegend=True,
        yaxis_ticksuffix=" kW",
        xaxis_ticksuffix=" timer",)
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': True, 'staticPlot': True})
    #st.info("Tips! Klikk på teksten i tegnforklaringen for å skru kurvene av/på.", icon="ℹ️")

start_time = time.time()
streamlit_settings(title="Energy Plan Zero, Østmarka", icon="h")
with st.sidebar:
    #c1, c2 = st.columns([1,1])
    st.image('src/img/trondheim-av-2.png', use_column_width = "auto")
    st.header("Energianalyse Østmarka")
    #with c1:
    #with c2:
    my_bar = st.progress(0, text="Tegn polygon")
COLUMN_1, COLUMN_2 = st.columns([3, 1])
###############
###############
MONTHS = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]
SPACEHEATING_COLOR = "#ff9966"
HISTORIC_COLOR = "#0000A3"
BEFORE_COLOR = "black"
RENEWABLE_COLOR = "green"
AFTER_COLOR = "#5C5CFF"
DHW_COLOR = "#b39200"
THERMAL_COLOR = "#48a23f"
ELECTRIC_COLOR = "blue"
ELSPECIFIC_COLOR = "#3399ff"
GRID_COLOR = "#7f7f7f"
STAND_OUT_COLOR = "#48a23f"
BASE_COLOR = "#1d3c34"
TOTAL_COLOR = "#1d3c34"
PRODUCED_HEAT_COLOR = "#1d3c34"
PRODUCED_EL_COLOR = "lightblue"
###############
###############
#with st.sidebar:
#    with st.expander("Simulering"):
#        if st.button("Kjør simulering"):
#           energy_analysis = EnergyAnalysis(
#                building_table = "building_table_østmarka_v2.xlsx",
#                energy_area_id = "energiomraadeid",
#                building_area_id = "bygningsomraadeid",
#                scenario_file_name = "input/scenarier.xlsx",
#                temperature_array_file_path = "input/utetemperatur.xlsx")
#           energy_analysis.main()
###############
###############
SCENARIO_NAMES = find_scenario_names("output")
#SCENARIO_NAMES = ['Referansesituasjon', 'Fjernvarme for Ringve VGS', 'Høyblokker med bergvarme', 'Solceller på alle tak']
selected_scenario_name = select_scenario()
df_position = read_position(f'output/{selected_scenario_name}')
#df_position = read_position(f'output/Referansesituasjon')
df_position = building_plan_filter(df_position)
SCENARIO_COMPARISON = scenario_comparison()
folium_map, gdf_buildings = create_map(df_position = df_position)
with COLUMN_1:
    st_map = display_map(folium_map)
with COLUMN_2:
    st.caption("Hvordan bruke Energy Plan Zero?")
    st.write(
        """Start med å tegne et polygon 
        for å gjøre et utvalg av bygg. 
        Da vises resultater for energi- og effekt 
        for utvalget.
        
        """)
    filtered_gdf = spatial_join(gdf_buildings)

object_ids = filtered_gdf['objectid'].astype(str)
object_ids['ID'] = 'ID'


results = {}
if SCENARIO_COMPARISON == False:
    SCENARIO_NAMES = [selected_scenario_name]

i = 0
increment = int(100/len(SCENARIO_NAMES))
for scenario_name in SCENARIO_NAMES:
    my_bar.progress(i, text = f"Laster inn {scenario_name}...")  
    df_hourly_data = read_hourly_data(object_ids, f"output/{scenario_name}")
    dict_arrays = get_dict_arrays(df_hourly_data)
    dict_months_sum = get_dict_months(dict_arrays, aggregation='sum')
    dict_months_max = get_dict_months(dict_arrays, aggregation='max')
    dict_months_average = get_dict_months(dict_arrays, aggregation='average')
    dict_sum = get_key_values(dict_months_sum, aggregation='sum')
    dict_max = get_key_values(dict_months_max, aggregation='max')

    results[scenario_name] = {
        'dict_arrays' : dict_arrays,
        'dict_months_sum' : dict_months_sum,
        'dict_months_max' : dict_months_max,
        'dict_months_average' : dict_months_average,
        'dict_sum' : dict_sum,
        'dict_max' : dict_max
    }
    i = i + increment
        
######################################################################
######################################################################
######################################################################
COLUMN_1, COLUMN_2 = st.columns([1, 1])    
with COLUMN_1:
    energy_effect_delivered_plot()
with COLUMN_2:
    energy_effect_scenario_plot()
my_bar.progress(int(i + (100 - i)/2), text = "Lager figurer...") 
######################################################################
if SCENARIO_COMPARISON == True:
    COLUMN_1, COLUMN_2 = st.columns([1, 3])
    with COLUMN_1:
        st.caption("Scenariosammenligning")
        st.write("Her vises en sammenligning av alle scenariene for det gjeldende utvalget.")
        #energy_effect_comparison_plot()
    with COLUMN_2:
        energy_effect_comparison_plot()
        #duration_curve_plot()
    
my_bar.progress(100, text="Fullført") 



#st.write(results)

#--





end_time = time.time()
#with st.sidebar:
#    st.title(f"Tidsbruk: {round((end_time - start_time),2)} sekunder")
