import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load data
shapefile_path = 'D:/UNI/Term 9/Project/Data/manhattan_zones.shp'
predicted_csv_path = 'D:/UNI/Term 9/Project/Data/forecast_pred.csv'

gdf = gpd.read_file(shapefile_path)
predicted_df = pd.read_csv(predicted_csv_path, parse_dates=['datetime'])

# Convert LID columns to integers
predicted_df.columns = ['datetime'] + [int(col.split('_')[1]) for col in predicted_df.columns[1:]]

# Title and description
st.title('Predicted Taxi Requests in Manhattan')
st.write('This project visualizes the predicted taxi requests in Manhattan for different days and hours. '
         'You can select a specific day and hour to see the predicted taxi requests on the map. '
         'The map uses a blue color scale where darker shades indicate higher demand.')

# Create a container for the layout
with st.container():
    col1, col2 = st.columns([1, 3], gap="large")

    with col1:
        st.markdown("### Select Day and Hour")
        day = st.selectbox('Select Day', ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
        hour = st.slider('Select Hour', 0, 23, 0)

    # Filter data based on selected day and hour
    predicted_df['day'] = predicted_df['datetime'].dt.day_name()
    predicted_df['hour'] = predicted_df['datetime'].dt.hour
    filtered_predicted_df = predicted_df[(predicted_df['day'] == day) & (predicted_df['hour'] == hour)]

    # Function to create choropleth map
    def create_choropleth(data, title):
        
        m = folium.Map(location=[40.7831, -73.9712], zoom_start=12)

        # Add choropleth layer
        folium.Choropleth(
            geo_data=gdf,
            name='choropleth',
            data=data,
            columns=['LocationID', 'value'],
            key_on='feature.properties.LocationID',
            fill_color='Blues',
            fill_opacity=1,
            line_opacity=0.5,
            legend_name=title
        ).add_to(m)
        
        # Add ID markers
        for _, r in gdf.iterrows():
            folium.Marker(
                location=[r['geometry'].centroid.y, r['geometry'].centroid.x],
                icon=folium.DivIcon(html=f"""<div style="font-size: 10pt; color : black; background-color: rgba(255, 255, 255, 0.7); padding: 2px;">{r['LocationID']}</div>""")
            ).add_to(m)
        
        return m

    # Create predicted map
    predicted_data = filtered_predicted_df.melt(id_vars=['datetime', 'day', 'hour'], var_name='LocationID', value_name='value')
    predicted_map = create_choropleth(predicted_data, 'Predicted Taxi Requests')

    with col2:
        st.subheader(f'{day} at {hour}:00')
        st.write('Predicted Data')
        st_folium(predicted_map, width=1200, height=700)
