import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load data
shapefile_path = "D:/UNI/Term 9/Project/Data/Raw/Zone/taxi_zones.shp"
predicted_csv_path = 'D:/UNI/Term 9/Project/Data/forecast_pred.csv'

gdf = gpd.read_file(shapefile_path)
predicted_df = pd.read_csv(predicted_csv_path, parse_dates=['datetime'])

# Title and description
st.title('Predicted Taxi Requests in Manhattan')
st.write('This project visualizes the predicted taxi requests in Manhattan for different days and hours. '
        'You can select a specific day and hour to see the predicted taxi requests on the map. '
        'The map uses a blue color scale where darker shades indicate higher demand.')

# Create a container for the layout
with st.container():
    col1, col2 = st.columns([1, 2], gap="medium")

    with col1:
        st.markdown("### Select Day and Hour")
        day = st.selectbox('Select Day', ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
        hour = st.slider('Select Hour', 0, 23, 0)

    # Filter data based on selected day and hour
    predicted_df['day'] = predicted_df['datetime'].dt.day_name()
    predicted_df['hour'] = predicted_df['datetime'].dt.hour
    filtered_predicted_df = predicted_df[(predicted_df['day'] == day) & (predicted_df['hour'] == hour)]

    # Function to create choropleth map
    def create_choropleth(test_df, ax, title):
        manhattan_data = test_df.filter(regex='LID_').sum().rename_axis('LocationID').reset_index()
        manhattan_data['LocationID'] = manhattan_data['LocationID'].str.extract('(\d+)').astype(int)
        manhattan_data.rename(columns={0: 'taxi_requests'}, inplace=True)

        manhattan_gdf = gdf[gdf['LocationID'].isin(manhattan_data['LocationID'])]
        manhattan_gdf = manhattan_gdf.merge(manhattan_data, on='LocationID')

        manhattan_gdf.plot(column='taxi_requests', ax=ax, cmap='Blues', edgecolor='black', linewidth=0.8)
        ax.set_facecolor('lightgray')
        ax.set_xticks([])
        ax.set_yticks([])

        for idx, row in manhattan_gdf.iterrows():
            ax.annotate(text=row['LocationID'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                        horizontalalignment='center', fontsize=4, color='black', weight='bold')

    # Create a figure with one subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Create choropleth map for predicted taxi requests volume
    create_choropleth(filtered_predicted_df, ax, 'Predicted Taxi Requests Volume')

    # Add a colorbar to the figure
    cax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(
        vmin=filtered_predicted_df.filter(regex='LID_').sum().min(), 
        vmax=filtered_predicted_df.filter(regex='LID_').sum().max()
    ))
    fig.colorbar(sm, cax=cax)

    # Display the plot in Streamlit
    with col2:
        st.subheader(f'{day} at {hour}:00')
        st.write('Predicted Data')
        st.pyplot(fig)
        
