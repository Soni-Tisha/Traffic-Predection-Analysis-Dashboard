import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="Traffic Flow Analysis & Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        text-align: center;
        color: #616161;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #9e9e9e;
        font-size: 0.8rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    # For demonstration purposes, let's create a sample dataframe similar to your data
    # In production, replace this with: df = pd.read_csv('your_data.csv')
    
    # Generate sample data similar to what you provided
    np.random.seed(42)
    date_range = pd.date_range(start='2024-01-01', periods=2000, freq='H')
    
    data = {
        'timestamp': date_range,
        'location_id': np.random.randint(1, 11, size=2000),
        'traffic_volume': np.random.randint(1000, 6000, size=2000),
        'avg_vehicle_speed': np.random.uniform(20, 80, size=2000),
        'vehicle_count_cars': np.random.randint(5, 50, size=2000),
        'vehicle_count_trucks': np.random.randint(1, 30, size=2000),
        'vehicle_count_bikes': np.random.randint(0, 15, size=2000),
        'weather_condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Foggy', 'Windy'], size=2000),
        'temperature': np.random.uniform(10, 35, size=2000),
        'humidity': np.random.uniform(35, 90, size=2000),
        'accident_reported': np.random.choice([0, 1], size=2000, p=[0.95, 0.05]),
        'signal_status': np.random.choice(['Red', 'Yellow', 'Green'], size=2000)
    }
    
    df = pd.DataFrame(data)
    
    # Extract time features
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['timestamp'].dt.month
    
    return df

# Function to predict traffic volume
def predict_traffic(df, location_id, features):
    # Filter data for the selected location
    location_data = df[df['location_id'] == location_id]
    
    # Prepare features and target
    X = location_data[features]
    y = location_data['traffic_volume']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_test, y_test, y_pred, model.feature_importances_

# Function to generate future data for prediction
def generate_future_data(df, location_id, hours_ahead=24):
    # Get the most recent timestamp
    latest_time = df['timestamp'].max()
    
    # Generate future timestamps
    future_times = pd.date_range(start=latest_time, periods=hours_ahead+1, freq='H')[1:]
    
    # Create a dataframe for future data
    future_data = pd.DataFrame()
    future_data['timestamp'] = future_times
    future_data['location_id'] = location_id
    
    # Extract time features
    future_data['hour'] = future_data['timestamp'].dt.hour
    future_data['day_of_week'] = future_data['timestamp'].dt.dayofweek
    future_data['is_weekend'] = future_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    future_data['month'] = future_data['timestamp'].dt.month
    
    # For weather features, we'll use the average values from the past day for the same location
    last_day_data = df[(df['location_id'] == location_id) & 
                       (df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=1))]
    
    if len(last_day_data) > 0:
        future_data['temperature'] = last_day_data['temperature'].mean()
        future_data['humidity'] = last_day_data['humidity'].mean()
        # For weather condition, use the most common from the past day
        future_data['weather_condition'] = last_day_data['weather_condition'].mode()[0]
    else:
        # Fallback to overall averages
        future_data['temperature'] = df['temperature'].mean()
        future_data['humidity'] = df['humidity'].mean()
        future_data['weather_condition'] = df['weather_condition'].mode()[0]
    
    # One-hot encode weather condition
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Foggy', 'Windy']
    for condition in weather_conditions:
        future_data[f'weather_{condition}'] = (future_data['weather_condition'] == condition).astype(int)
    
    return future_data

# Function to calculate traffic metrics
def calculate_metrics(df):
    metrics = {}
    
    # Overall metrics
    metrics['total_volume'] = df['traffic_volume'].sum()
    metrics['avg_speed'] = df['avg_vehicle_speed'].mean()
    metrics['total_vehicles'] = df['vehicle_count_cars'].sum() + df['vehicle_count_trucks'].sum() + df['vehicle_count_bikes'].sum()
    metrics['accident_count'] = df['accident_reported'].sum()
    metrics['peak_hour'] = df.groupby('hour')['traffic_volume'].mean().idxmax()
    
    # Peak time metrics
    morning_peak = df[(df['hour'] >= 7) & (df['hour'] <= 9)]
    evening_peak = df[(df['hour'] >= 16) & (df['hour'] <= 18)]
    
    metrics['morning_peak_volume'] = morning_peak['traffic_volume'].mean()
    metrics['evening_peak_volume'] = evening_peak['traffic_volume'].mean()
    
    # Weather impact
    weather_impact = df.groupby('weather_condition')['traffic_volume'].mean().to_dict()
    metrics['weather_impact'] = weather_impact
    
    # Location with highest traffic
    location_traffic = df.groupby('location_id')['traffic_volume'].sum()
    metrics['busiest_location'] = location_traffic.idxmax()
    metrics['busiest_location_volume'] = location_traffic.max()
    
    return metrics

# Main function
def main():
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.markdown("<h2 style='text-align: center;'>🚦 Dashboard Controls</h2>", unsafe_allow_html=True)
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Traffic Analysis", "Predictive Analytics", "Traffic Patterns", "Real-time Monitor"]
    )
    
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
    # Filters in sidebar
    st.sidebar.markdown("<h3>Filters</h3>", unsafe_allow_html=True)
    
    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_df = df[mask]
        print("***************",filtered_df)
    else:
        filtered_df = df
    
    selected_weekday = (filtered_df['is_weekend'] == 0).any()
    selected_weekend = (filtered_df['is_weekend'] == 1).any()
    # Location filter
    locations = sorted(df['location_id'].unique())
    selected_locations = st.sidebar.multiselect(
        "Select Locations",
        options=locations,
        default=locations[:3]  # Default to first 3 locations
    )
    
    if selected_locations:
        filtered_df = filtered_df[filtered_df['location_id'].isin(selected_locations)]
    
    # Weather condition filter
    weather_conditions = sorted(df['weather_condition'].unique())
    selected_weather = st.sidebar.multiselect(
        "Weather Conditions",
        options=weather_conditions,
        default=weather_conditions
    )
    
    if selected_weather:
        filtered_df = filtered_df[filtered_df['weather_condition'].isin(selected_weather)]
    
    # Time of day filter
    hour_range = st.sidebar.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )
    
    filtered_df = filtered_df[(filtered_df['hour'] >= hour_range[0]) & (filtered_df['hour'] <= hour_range[1])]
    
    # Calculate metrics based on filtered data
    metrics = calculate_metrics(filtered_df)
    
    # Page content
    if page == "Overview":
        st.markdown("<h1 class='main-header'>🚦 Traffic Flow Analysis & Prediction</h1>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Comprehensive dashboard for analyzing traffic patterns and predicting future conditions</p>", unsafe_allow_html=True)
        
        # Display key metrics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Total Traffic Volume</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{metrics['total_volume']:,.0f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Average Speed (mph)</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{metrics['avg_speed']:.1f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Accidents Reported</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{metrics['accident_count']}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Traffic volume by hour
        st.markdown("<h2 class='sub-header'>Traffic Volume by Hour of Day</h2>", unsafe_allow_html=True)
        
        hourly_traffic = filtered_df.groupby('hour')['traffic_volume'].mean().reset_index()
        
        fig = px.line(
            hourly_traffic,
            x='hour',
            y='traffic_volume',
            markers=True,
            title=None,
            labels={'hour': 'Hour of Day', 'traffic_volume': 'Average Traffic Volume'}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Hour of Day',
            yaxis_title='Average Traffic Volume'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Traffic by location and weather
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h2 class='sub-header'>Traffic by Location</h2>", unsafe_allow_html=True)
            
            location_traffic = filtered_df.groupby('location_id')['traffic_volume'].sum().reset_index()
            location_traffic = location_traffic.sort_values('traffic_volume', ascending=False)
            
            fig = px.bar(
                location_traffic,
                x='location_id',
                y='traffic_volume',
                title=None,
                labels={'location_id': 'Location ID', 'traffic_volume': 'Total Traffic Volume'}
            )
            
            fig.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h2 class='sub-header'>Traffic by Weather Condition</h2>", unsafe_allow_html=True)
            
            weather_traffic = filtered_df.groupby('weather_condition')['traffic_volume'].mean().reset_index()
            weather_traffic = weather_traffic.sort_values('traffic_volume', ascending=False)
            
            fig = px.bar(
                weather_traffic,
                x='weather_condition',
                y='traffic_volume',
                title=None,
                color='weather_condition',
                labels={'weather_condition': 'Weather Condition', 'traffic_volume': 'Average Traffic Volume'}
            )
            
            fig.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Vehicle composition
        st.markdown("<h2 class='sub-header'>Vehicle Composition</h2>", unsafe_allow_html=True)
        
        vehicle_data = {
            'Vehicle Type': ['Cars', 'Trucks', 'Bikes'],
            'Count': [
                filtered_df['vehicle_count_cars'].sum(),
                filtered_df['vehicle_count_trucks'].sum(),
                filtered_df['vehicle_count_bikes'].sum()
            ]
        }
        
        vehicle_df = pd.DataFrame(vehicle_data)
        
        fig = px.pie(
            vehicle_df,
            values='Count',
            names='Vehicle Type',
            title=None,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent trends
        st.markdown("<h2 class='sub-header'>Recent Traffic Trends</h2>", unsafe_allow_html=True)
        
        # Group by date and calculate daily average
        daily_traffic = filtered_df.groupby('date')['traffic_volume'].mean().reset_index()
        
        fig = px.line(
            daily_traffic,
            x='date',
            y='traffic_volume',
            markers=True,
            title=None,
            labels={'date': 'Date', 'traffic_volume': 'Average Daily Traffic Volume'}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Traffic Analysis":
        st.markdown("<h1 class='main-header'>Traffic Analysis Dashboard</h1>", unsafe_allow_html=True)
        
        # Advanced analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Correlation Analysis", "Peak Hours Analysis", "Weather Impact", "Vehicle Distribution"]
        )
        
        if analysis_type == "Correlation Analysis":
            st.markdown("<h2 class='sub-header'>Correlation Between Traffic Variables</h2>", unsafe_allow_html=True)
            
            # Select variables for correlation
            corr_vars = st.multiselect(
                "Select Variables for Correlation Analysis",
                options=['traffic_volume', 'avg_vehicle_speed', 'vehicle_count_cars', 'vehicle_count_trucks', 
                         'vehicle_count_bikes', 'temperature', 'humidity', 'hour', 'is_weekend'],
                default=['traffic_volume', 'avg_vehicle_speed', 'temperature', 'hour']
            )
            
            if len(corr_vars) < 2:
                st.warning("Please select at least two variables for correlation analysis.")
            else:
                # Calculate correlation matrix
                corr_matrix = filtered_df[corr_vars].corr()
                
                # Plot correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                plt.title("Correlation Matrix")
                st.pyplot(fig)
                
                # Provide insights
                st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                st.markdown("<h3>Key Insights:</h3>", unsafe_allow_html=True)
                
                # Identify strongest correlations
                corr_pairs = []
                for i in range(len(corr_vars)):
                    for j in range(i+1, len(corr_vars)):
                        corr_value = corr_matrix.iloc[i, j]
                        corr_pairs.append((corr_vars[i], corr_vars[j], corr_value))
                
                # Sort by absolute correlation value
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Display top correlations
                for var1, var2, corr_val in corr_pairs[:3]:
                    direction = "positive" if corr_val > 0 else "negative"
                    strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.3 else "weak"
                    
                    st.markdown(f"• {var1} and {var2} have a {strength} {direction} correlation ({corr_val:.2f})", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Scatter plot of two variables
                st.markdown("<h3>Detailed Relationship Analysis</h3>", unsafe_allow_html=True)
                
                x_var = st.selectbox("Select X-axis Variable", options=corr_vars, index=0)
                y_var = st.selectbox("Select Y-axis Variable", 
                                   options=[v for v in corr_vars if v != x_var], 
                                   index=min(1, len(corr_vars)-1))
                
                fig = px.scatter(
                    filtered_df, 
                    x=x_var, 
                    y=y_var,
                    color='weather_condition',
                    trendline="ols",
                    title=f"Relationship between {x_var} and {y_var}",
                    labels={x_var: x_var.replace('_', ' ').title(), y_var: y_var.replace('_', ' ').title()}
                )
                
                fig.update_layout(
                    height=500,
                    margin=dict(t=50, b=50, l=50, r=50),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Peak Hours Analysis":
            st.markdown("<h2 class='sub-header'>Peak Hours Traffic Analysis</h2>", unsafe_allow_html=True)
            
            # Get hourly traffic data
            hourly_traffic = filtered_df.groupby('hour')['traffic_volume'].mean().reset_index()
            
            # Plot hourly traffic
            fig = px.line(
                hourly_traffic,
                x='hour',
                y='traffic_volume',
                markers=True,
                title="Traffic Volume by Hour of Day",
                labels={'hour': 'Hour of Day', 'traffic_volume': 'Average Traffic Volume'}
            )
            
            # Add bands for peak hours
            fig.add_vrect(
                x0=7, x1=9,
                fillcolor="rgba(255, 0, 0, 0.1)", opacity=0.5,
                layer="below", line_width=0,
                annotation_text="Morning Peak",
                annotation_position="top right"
            )
            
            fig.add_vrect(
                x0=16, x1=18,
                fillcolor="rgba(0, 0, 255, 0.1)", opacity=0.5,
                layer="below", line_width=0,
                annotation_text="Evening Peak",
                annotation_position="top right"
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=50, b=50, l=50, r=50),
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak hours statistics
            morning_peak = filtered_df[(filtered_df['hour'] >= 7) & (filtered_df['hour'] <= 9)]
            evening_peak = filtered_df[(filtered_df['hour'] >= 16) & (filtered_df['hour'] <= 18)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Morning Peak (7-9 AM)</h3>", unsafe_allow_html=True)
                
                
                
                # Most common weather during morning peak
                if not morning_peak.empty:

                    st.metric("Average Traffic Volume", f"{morning_peak['traffic_volume'].mean():.1f}")
                    st.metric("Average Speed", f"{morning_peak['avg_vehicle_speed'].mean():.1f} mph")
                    st.metric("Accidents", f"{morning_peak['accident_reported'].sum()}")

                    morning_weather = morning_peak['weather_condition'].value_counts().idxmax()
                    st.metric("Most Common Weather", morning_weather)
                else:
                   # st.warning("No traffic data available during Morning Peak hours.")
                    st.metric("Average Traffic Volume", "0")
                    st.metric("Average Speed", "0")
                    st.metric("Accidents", "0")
                    st.metric("Most Common Weather", "0")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Evening Peak (4-6 PM)</h3>", unsafe_allow_html=True)
                
                
                # Most common weather during evening peak
                if not evening_peak.empty:

                    st.metric("Average Traffic Volume", f"{evening_peak['traffic_volume'].mean():.1f}")
                    st.metric("Average Speed", f"{evening_peak['avg_vehicle_speed'].mean():.1f} mph")
                    st.metric("Accidents", f"{evening_peak['accident_reported'].sum()}")
                

                    evening_weather = evening_peak['weather_condition'].value_counts().idxmax()
                    st.metric("Most Common Weather", evening_weather)
                else:
                    #st.warning("No traffic data available during Evening Peak hours.")
                    st.metric("Average Traffic Volume", "0")
                    st.metric("Average Speed", "-")
                    st.metric("Accidents", "0")
                    st.metric("Most Common Weather", "0")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Peak analysis by location
            st.markdown("<h3>Peak Hour Traffic by Location</h3>", unsafe_allow_html=True)
            
            # Get peak hour traffic by location
            peak_location = filtered_df[
                ((filtered_df['hour'] >= 7) & (filtered_df['hour'] <= 9)) | 
                ((filtered_df['hour'] >= 16) & (filtered_df['hour'] <= 18))
            ]
            
            peak_location['peak_type'] = 'Off-Peak'
            peak_location.loc[(peak_location['hour'] >= 7) & (peak_location['hour'] <= 9), 'peak_type'] = 'Morning Peak'
            peak_location.loc[(peak_location['hour'] >= 16) & (peak_location['hour'] <= 18), 'peak_type'] = 'Evening Peak'
            
            peak_location_data = peak_location.groupby(['location_id', 'peak_type'])['traffic_volume'].mean().reset_index()
            
            fig = px.bar(
                peak_location_data,
                x='location_id',
                y='traffic_volume',
                color='peak_type',
                barmode='group',
                title="Peak Hour Traffic by Location",
                labels={'location_id': 'Location ID', 'traffic_volume': 'Average Traffic Volume', 'peak_type': 'Time Period'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=50, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Weather Impact":
            st.markdown("<h2 class='sub-header'>Weather Impact Analysis</h2>", unsafe_allow_html=True)
            
            # Calculate average traffic by weather condition
            weather_impact = filtered_df.groupby('weather_condition').agg({
                'traffic_volume': 'mean',
                'avg_vehicle_speed': 'mean',
                'accident_reported': 'sum'
            }).reset_index()
            
            # Plot traffic volume by weather
            fig = px.bar(
                weather_impact,
                x='weather_condition',
                y='traffic_volume',
                color='weather_condition',
                title="Average Traffic Volume by Weather Condition",
                labels={'weather_condition': 'Weather Condition', 'traffic_volume': 'Average Traffic Volume'}
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=50, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed weather impact statistics
            st.markdown("<h3>Detailed Weather Impact Statistics</h3>", unsafe_allow_html=True)
            
            # Format the data for display
            weather_impact['traffic_volume'] = weather_impact['traffic_volume'].round(1)
            weather_impact['avg_vehicle_speed'] = weather_impact['avg_vehicle_speed'].round(1)
            weather_impact.rename(columns={
                'traffic_volume': 'Avg Traffic Volume',
                'avg_vehicle_speed': 'Avg Speed (mph)',
                'accident_reported': 'Accidents',
                'weather_condition': 'Weather Condition'
            }, inplace=True)
            
            st.dataframe(weather_impact, use_container_width=True)
            
            # Weather vs. Speed
            st.markdown("<h3>Weather Impact on Vehicle Speed</h3>", unsafe_allow_html=True)
            
            fig = px.box(
                filtered_df,
                x='weather_condition',
                y='avg_vehicle_speed',
                color='weather_condition',
                title="Vehicle Speed Distribution by Weather Condition",
                labels={'weather_condition': 'Weather Condition', 'avg_vehicle_speed': 'Average Vehicle Speed (mph)'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=50, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weather vs. Accidents
            st.markdown("<h3>Weather Impact on Accident Rates</h3>", unsafe_allow_html=True)
            
            # Calculate accident rates by weather
            accidents_by_weather = filtered_df.groupby('weather_condition').agg({
                'accident_reported': 'sum',
                'timestamp': 'count'
            }).reset_index()
            
            accidents_by_weather['accident_rate'] = (accidents_by_weather['accident_reported'] / 
                                                    accidents_by_weather['timestamp']) * 100
            
            fig = px.bar(
                accidents_by_weather,
                x='weather_condition',
                y='accident_rate',
                color='weather_condition',
                title="Accident Rate by Weather Condition (%)",
                labels={'weather_condition': 'Weather Condition', 'accident_rate': 'Accident Rate (%)'}
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=50, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide insights
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Weather Insights:</h3>", unsafe_allow_html=True)
            
            # Find weather with highest/lowest traffic
            highest_traffic_weather = weather_impact.loc[weather_impact['Avg Traffic Volume'].idxmax()]['Weather Condition']
            lowest_traffic_weather = weather_impact.loc[weather_impact['Avg Traffic Volume'].idxmin()]['Weather Condition']
            
            # Find weather with highest/lowest speed
            highest_speed_weather = weather_impact.loc[weather_impact['Avg Speed (mph)'].idxmax()]['Weather Condition']
            lowest_speed_weather = weather_impact.loc[weather_impact['Avg Speed (mph)'].idxmin()]['Weather Condition']
            
            # Find weather with highest accident rate
            highest_accident_weather = accidents_by_weather.loc[accidents_by_weather['accident_rate'].idxmax()]['weather_condition']
            
            st.markdown(f"• Highest traffic volume occurs during **{highest_traffic_weather}** weather", unsafe_allow_html=True)
            st.markdown(f"• Lowest traffic volume occurs during **{lowest_traffic_weather}** weather", unsafe_allow_html=True)
            st.markdown(f"• Vehicles move fastest during **{highest_speed_weather}** weather", unsafe_allow_html=True)
            st.markdown(f"• Vehicles move slowest during **{lowest_speed_weather}** weather", unsafe_allow_html=True)
            st.markdown(f"• The highest accident rate occurs during **{highest_accident_weather}** weather", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif analysis_type == "Vehicle Distribution":
            st.markdown("<h2 class='sub-header'>Vehicle Distribution Analysis</h2>", unsafe_allow_html=True)
            
            # Aggregate vehicle counts
            vehicle_counts = pd.DataFrame({
                'Cars': filtered_df['vehicle_count_cars'].sum(),
                'Trucks': filtered_df['vehicle_count_trucks'].sum(),
                'Bikes': filtered_df['vehicle_count_bikes'].sum()
            }, index=['Count'])
            
            vehicle_counts = vehicle_counts.T.reset_index()
            vehicle_counts.columns = ['Vehicle Type', 'Count']
            
            # Plot vehicle distribution
            fig = px.pie(
                vehicle_counts,
                values='Count',
                names='Vehicle Type',
                title="Overall Vehicle Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Vehicle distribution by hour
            st.markdown("<h3>Vehicle Distribution by Hour</h3>", unsafe_allow_html=True)
            
            hourly_vehicles = filtered_df.groupby('hour').agg({
                'vehicle_count_cars': 'mean',
                'vehicle_count_trucks': 'mean',
                'vehicle_count_bikes': 'mean'
            }).reset_index()
            
            hourly_vehicles_melted = pd.melt(
                hourly_vehicles,
                id_vars=['hour'],
                value_vars=['vehicle_count_cars', 'vehicle_count_trucks', 'vehicle_count_bikes'],
                var_name='Vehicle Type',
                value_name='Average Count'
            )
            
            hourly_vehicles_melted['Vehicle Type'] = hourly_vehicles_melted['Vehicle Type'].map({
                'vehicle_count_cars': 'Cars',
                'vehicle_count_trucks': 'Trucks',
                'vehicle_count_bikes': 'Bikes'
            })
            
            fig = px.line(
                hourly_vehicles_melted,
                x='hour',
                y='Average Count',
                color='Vehicle Type',
                markers=True,
                title="Average Vehicle Count by Hour of Day",
                labels={'hour': 'Hour of Day', 'Average Count': 'Average Vehicle Count'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=50, b=50, l=50, r=50),
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Vehicle distribution by weather
            st.markdown("<h3>Vehicle Distribution by Weather</h3>", unsafe_allow_html=True)
            
            weather_vehicles = filtered_df.groupby('weather_condition').agg({
                'vehicle_count_cars': 'mean',
                'vehicle_count_trucks': 'mean',
                'vehicle_count_bikes': 'mean'
            }).reset_index()
            
            weather_vehicles_melted = pd.melt(
                weather_vehicles,
                id_vars=['weather_condition'],
                value_vars=['vehicle_count_cars', 'vehicle_count_trucks', 'vehicle_count_bikes'],
                var_name='Vehicle Type',
                value_name='Average Count'
            )
            
            weather_vehicles_melted['Vehicle Type'] = weather_vehicles_melted['Vehicle Type'].map({
                'vehicle_count_cars': 'Cars',
                'vehicle_count_trucks': 'Trucks',
                'vehicle_count_bikes': 'Bikes'
            })
            
            fig = px.bar(
                weather_vehicles_melted,
                x='weather_condition',
                y='Average Count',
                color='Vehicle Type',
                title="Average Vehicle Count by Weather Condition",
                labels={'weather_condition': 'Weather Condition', 'Average Count': 'Average Vehicle Count'},
                barmode='group'
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=50, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights about vehicle distribution
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Vehicle Distribution Insights:</h3>", unsafe_allow_html=True)
            
            # Calculate bike percentage in good vs. bad weather
            good_weather = filtered_df[filtered_df['weather_condition'].isin(['Sunny', 'Cloudy'])]
            bad_weather = filtered_df[filtered_df['weather_condition'].isin(['Rainy', 'Foggy', 'Windy'])]
            
            good_weather_bikes_pct = (good_weather['vehicle_count_bikes'].sum() / 
                                     (good_weather['vehicle_count_cars'].sum() + 
                                      good_weather['vehicle_count_trucks'].sum() + 
                                      good_weather['vehicle_count_bikes'].sum())) * 100
            
            bad_weather_bikes_pct = (bad_weather['vehicle_count_bikes'].sum() / 
                                    (bad_weather['vehicle_count_cars'].sum() + 
                                     bad_weather['vehicle_count_trucks'].sum() + 
                                     bad_weather['vehicle_count_bikes'].sum())) * 100
            
            # Calculate peak hour vehicle distribution
            peak_hours = filtered_df[(filtered_df['hour'] >= 7) & (filtered_df['hour'] <= 9) | 
                                   (filtered_df['hour'] >= 16) & (filtered_df['hour'] <= 18)]
            
            peak_cars_pct = (peak_hours['vehicle_count_cars'].sum() / 
                            (peak_hours['vehicle_count_cars'].sum() + 
                             peak_hours['vehicle_count_trucks'].sum() + 
                             peak_hours['vehicle_count_bikes'].sum())) * 100
            
            # Insights
            st.markdown(f"• Cars make up **{peak_cars_pct:.1f}%** of traffic during peak hours", unsafe_allow_html=True)
            st.markdown(f"• Bike usage drops from **{good_weather_bikes_pct:.1f}%** in good weather to **{bad_weather_bikes_pct:.1f}%** in bad weather", unsafe_allow_html=True)
            
            # Calculate when trucks are most common
            truck_by_hour = filtered_df.groupby('hour')['vehicle_count_trucks'].mean()
            max_truck_hour = truck_by_hour.idxmax()
            
            st.markdown(f"• Truck traffic peaks at **{max_truck_hour}:00 hours**", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    elif page == "Predictive Analytics":
        st.markdown("<h1 class='main-header'>Traffic Prediction Dashboard</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <p style='font-size: 1.1rem;'>This tool uses machine learning to predict future traffic patterns based on historical data and conditions.</p>
        """, unsafe_allow_html=True)
        
        # Location selector for prediction
        pred_location = st.selectbox(
            "Select Location for Prediction",
            options=sorted(df['location_id'].unique()),
            index=0
        )
        
        # Define features for prediction
        features = ['hour', 'day_of_week', 'is_weekend', 'month', 'temperature', 'humidity']
        
        # Add weather condition as one-hot encoded features
        weather_conditions = sorted(df['weather_condition'].unique())
        for condition in weather_conditions:
            df[f'weather_{condition}'] = (df['weather_condition'] == condition).astype(int)
            features.append(f'weather_{condition}')
        
        # Train model and make predictions
        model, mse, r2, X_test, y_test, y_pred, feature_importance = predict_traffic(df, pred_location, features)
        
        # Display model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Model R² Score</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{r2:.3f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Mean Squared Error</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{mse:.1f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Plot actual vs predicted
        st.markdown("<h2 class='sub-header'>Actual vs. Predicted Traffic Volume</h2>", unsafe_allow_html=True)
        
        # Create a dataframe for plotting
        pred_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        fig = px.scatter(
            pred_df,
            x='Actual',
            y='Predicted',
            title=None,
            labels={'Actual': 'Actual Traffic Volume', 'Predicted': 'Predicted Traffic Volume'}
        )
        
        # Add perfect prediction line
        fig.add_trace(
            go.Scatter(
                x=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                y=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            height=500,
            margin=dict(t=20, b=50, l=50, r=50),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
        
        # Create feature importance dataframe
        fi_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        })
        
        fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=None,
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature'}
        )
        
        fig.update_layout(
            height=500,
            margin=dict(t=20, b=50, l=50, r=50),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.markdown("<h2 class='sub-header'>Future Traffic Predictions</h2>", unsafe_allow_html=True)
        
        # Generate future data for prediction
        hours_ahead = st.slider("Hours to Predict Ahead", min_value=6, max_value=72, value=24, step=6)
        
        future_data = generate_future_data(df, pred_location, hours_ahead)
        
        # Prepare features for prediction
        X_future = future_data[features]
        
        # Make predictions
        future_predictions = model.predict(X_future)
        
        # Create future predictions dataframe
        future_pred_df = pd.DataFrame({
            'Timestamp': future_data['timestamp'],
            'Hour': future_data['hour'],
            'Traffic Volume': future_predictions
        })
        
        # Plot future predictions
        fig = px.line(
            future_pred_df,
            x='Timestamp',
            y='Traffic Volume',
            markers=True,
            title=None,
            labels={'Timestamp': 'Time', 'Traffic Volume': 'Predicted Traffic Volume'}
        )
        
        fig.update_layout(
            height=500,
            margin=dict(t=20, b=50, l=50, r=50),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Traffic trend by hour for the future period
        st.markdown("<h3>Predicted Traffic by Hour</h3>", unsafe_allow_html=True)
        
        hourly_prediction = future_pred_df.groupby('Hour')['Traffic Volume'].mean().reset_index()
        
        fig = px.bar(
            hourly_prediction,
            x='Hour',
            y='Traffic Volume',
            title=None,
            labels={'Hour': 'Hour of Day', 'Traffic Volume': 'Average Predicted Traffic'},
            color='Traffic Volume',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=20, b=50, l=50, r=50),
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download predictions
        st.markdown("<h3>Download Predictions</h3>", unsafe_allow_html=True)
        
        # Format the future predictions for download
        future_pred_df['Timestamp'] = future_pred_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        future_pred_df['Traffic Volume'] = future_pred_df['Traffic Volume'].round(2)
        
        csv = future_pred_df.to_csv(index=False)
        
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"traffic_predictions_location_{pred_location}.csv",
            mime="text/csv"
        )
        
    elif page == "Traffic Patterns":
        st.markdown("<h1 class='main-header'>Traffic Pattern Analysis</h1>", unsafe_allow_html=True)
        
        pattern_type = st.selectbox(
            "Select Pattern Analysis Type",
            ["Weekday vs Weekend", "Monthly Trends", "Signal Status Impact", "Accident Analysis"]
        )
        
        if pattern_type == "Weekday vs Weekend":
            st.markdown("<h2 class='sub-header'>Weekday vs Weekend Traffic Patterns</h2>", unsafe_allow_html=True)
            
            # Calculate weekday vs weekend traffic
            if selected_weekday:
                weekday_data = filtered_df[filtered_df['is_weekend'] == 0]
                print("----------------",weekday_data)
            else:
                weekday_data = pd.DataFrame()  # Empty DataFrame if no weekday data
                print("^^^^^^^^^^^^^^^6",weekday_data)

            if selected_weekend:
                weekend_data = filtered_df[filtered_df['is_weekend'] == 1]
            else:
                weekend_data = pd.DataFrame() 
            
            # Hourly average by day type
            if not weekday_data.empty:
                weekday_hourly = weekday_data.groupby('hour')['traffic_volume'].mean().reset_index()
                weekday_hourly['Day Type'] = 'Weekday'
            else:
                weekday_hourly = pd.DataFrame(columns=['hour', 'traffic_volume', 'Day Type'])
            
            if not weekend_data.empty:
                weekend_hourly = weekend_data.groupby('hour')['traffic_volume'].mean().reset_index()
                weekend_hourly['Day Type'] = 'Weekend'
            else:
                weekend_hourly = pd.DataFrame(columns=['hour', 'traffic_volume', 'Day Type'])

            day_type_df = pd.concat([weekday_hourly, weekend_hourly])
            
            # Plot weekday vs weekend traffic
            fig = px.line(
                day_type_df,
                x='hour',
                y='traffic_volume',
                color='Day Type',
                markers=True,
                title=None,
                labels={'hour': 'Hour of Day', 'traffic_volume': 'Average Traffic Volume', 'Day Type': 'Day Type'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=20, b=50, l=50, r=50),
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                 if not weekday_data.empty:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Weekday Statistics</h3>", unsafe_allow_html=True)
                    
                    st.metric("Average Traffic Volume", f"{weekday_data['traffic_volume'].mean():.1f}" if not pd.isna(weekday_data['traffic_volume'].mean()) else 0)
                    st.metric("Average Speed", f"{weekday_data['avg_vehicle_speed'].mean():.1f} mph" if not pd.isna(weekday_data['avg_vehicle_speed'].mean()) else 0)
                    st.metric("Peak Hour", f"{weekday_hourly.loc[weekday_hourly['traffic_volume'].idxmax(), 'hour']}:00" if not weekday_hourly.empty else 'N/A')
                    
                    weekday_morning_peak = weekday_data[(weekday_data['hour'] >= 7) & (weekday_data['hour'] <= 9)]['traffic_volume'].mean()  if not weekday_data.empty else 0
                    weekday_evening_peak = weekday_data[(weekday_data['hour'] >= 16) & (weekday_data['hour'] <= 18)]['traffic_volume'].mean() if not weekday_data.empty else 0

                    weekday_morning_peak = 0 if pd.isna(weekday_morning_peak) else weekday_morning_peak
                    weekday_evening_peak = 0 if pd.isna(weekday_evening_peak) else weekday_evening_peak
                    
                    st.metric("Morning Peak (7-9 AM)", f"{weekday_morning_peak:.1f}" if weekday_morning_peak != 0 else 0)
                    st.metric("Evening Peak (4-6 PM)", f"{weekday_evening_peak:.1f}" if weekday_evening_peak != 0 else 0)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                if not weekend_data.empty:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Weekend Statistics</h3>", unsafe_allow_html=True)
                    
                    st.metric("Average Traffic Volume", f"{weekend_data['traffic_volume'].mean():.1f}" if not pd.isna(weekend_data['traffic_volume'].mean()) else 0)
                    st.metric("Average Speed", f"{weekend_data['avg_vehicle_speed'].mean():.1f} mph" if not pd.isna(weekend_data['avg_vehicle_speed'].mean()) else 0)
                    st.metric("Peak Hour", f"{weekend_hourly.loc[weekend_hourly['traffic_volume'].idxmax(), 'hour']}:00" if not weekend_hourly.empty else 'N/A')
                    
                    weekend_morning_peak = weekend_data[(weekend_data['hour'] >= 7) & (weekend_data['hour'] <= 9)]['traffic_volume'].mean() if not weekend_data.empty else 0
                    weekend_evening_peak = weekend_data[(weekend_data['hour'] >= 16) & (weekend_data['hour'] <= 18)]['traffic_volume'].mean() if not weekend_data.empty else 0

                    weekend_morning_peak = 0 if pd.isna(weekend_morning_peak) else weekend_morning_peak
                    weekend_evening_peak = 0 if pd.isna(weekend_evening_peak) else weekend_evening_peak
        
                    
                    st.metric("Morning Peak (7-9 AM)", f"{weekend_morning_peak:.1f}" if weekend_morning_peak != 0 else 0)
                    st.metric("Evening Peak (4-6 PM)", f"{weekend_evening_peak:.1f}" if weekend_evening_peak != 0 else 0)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Vehicle distribution comparison
            st.markdown("<h3>Vehicle Distribution: Weekday vs Weekend</h3>", unsafe_allow_html=True)


            weekday_car_count = weekday_data['vehicle_count_cars'].sum() if 'vehicle_count_cars' in weekday_data.columns else 0
            weekday_truck_count = weekday_data['vehicle_count_trucks'].sum() if 'vehicle_count_trucks' in weekday_data.columns else 0
            weekday_bike_count = weekday_data['vehicle_count_bikes'].sum() if 'vehicle_count_bikes' in weekday_data.columns else 0

            weekend_car_count = weekend_data['vehicle_count_cars'].sum() if 'vehicle_count_cars' in weekend_data.columns else 0
            weekend_truck_count = weekend_data['vehicle_count_trucks'].sum() if 'vehicle_count_trucks' in weekend_data.columns else 0
            weekend_bike_count = weekend_data['vehicle_count_bikes'].sum() if 'vehicle_count_bikes' in weekend_data.columns else 0

            weekday_total = weekday_car_count + weekday_truck_count + weekday_bike_count
            weekend_total = weekend_car_count + weekend_truck_count + weekend_bike_count
            
            # Calculate vehicle proportions
           # weekday_total = (weekday_data['vehicle_count_cars'].sum() + 
            #                weekday_data['vehicle_count_trucks'].sum() + 
             #               weekday_data['vehicle_count_bikes'].sum())
            #print("$$$$$$$$$$$4",weekday_total)
            
            #weekend_total = (weekend_data['vehicle_count_cars'].sum() + 
             #               weekend_data['vehicle_count_trucks'].sum() + 
              #              weekend_data['vehicle_count_bikes'].sum())
            #print("%%%%%%%%%%%%%%%%",weekend_total)
            
            vehicle_dist_data = pd.DataFrame({
                'Day Type': ['Weekday', 'Weekday', 'Weekday', 'Weekend', 'Weekend', 'Weekend'],
                'Vehicle Type': ['Cars', 'Trucks', 'Bikes', 'Cars', 'Trucks', 'Bikes'],
                'Count': [
                    weekday_car_count,
                    weekday_truck_count,
                    weekday_bike_count,
                    weekend_car_count,
                    weekend_truck_count,
                    weekend_bike_count
                ],
                'Percentage': [
                    (weekday_car_count / weekday_total) * 100 if weekday_total > 0 else 0,
                    (weekday_truck_count / weekday_total) * 100 if weekday_total > 0 else 0,
                    (weekday_bike_count / weekday_total) * 100 if weekday_total > 0 else 0,
                    (weekend_car_count / weekend_total) * 100 if weekend_total > 0 else 0,
                    (weekend_truck_count / weekend_total) * 100 if weekend_total > 0 else 0,
                    (weekend_bike_count / weekend_total) * 100 if weekend_total > 0 else 0
                ]
            })
            
            fig = px.bar(
                vehicle_dist_data,
                x='Day Type',
                y='Percentage',
                color='Vehicle Type',
                barmode='group',
                title=None,
                labels={'Day Type': 'Day Type', 'Percentage': 'Percentage (%)', 'Vehicle Type': 'Vehicle Type'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Insights:</h3>", unsafe_allow_html=True)
            
            # Calculate percentage difference in traffic volume
            if not weekend_data.empty and not weekday_data.empty:
                pct_diff = ((weekday_data['traffic_volume'].mean() - weekend_data['traffic_volume'].mean()) / 
                            weekend_data['traffic_volume'].mean()) * 100
            
                # Calculate peak hour difference
                weekday_peak = weekday_hourly.loc[weekday_hourly['traffic_volume'].idxmax(), 'hour']
                weekend_peak = weekend_hourly.loc[weekend_hourly['traffic_volume'].idxmax(), 'hour']
                
                st.markdown(f"• Weekday traffic is **{abs(pct_diff):.1f}%** {'higher' if pct_diff > 0 else 'lower'} than weekend traffic", unsafe_allow_html=True)
                st.markdown(f"• Weekday peak traffic occurs at **{weekday_peak}:00** hours", unsafe_allow_html=True)
                st.markdown(f"• Weekend peak traffic occurs at **{weekend_peak}:00** hours", unsafe_allow_html=True)
                
                # Compare morning and evening peaks
                weekday_peak_ratio = weekday_evening_peak / weekday_morning_peak if weekday_morning_peak > 0 else 0
                weekend_peak_ratio = weekend_evening_peak / weekend_morning_peak if weekend_morning_peak > 0 else 0
                
                st.markdown(f"• Evening-to-morning peak ratio: **{weekday_peak_ratio:.2f}** (weekday) vs **{weekend_peak_ratio:.2f}** (weekend)", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif pattern_type == "Monthly Trends":
            st.markdown("<h2 class='sub-header'>Monthly Traffic Trends</h2>", unsafe_allow_html=True)
            
            # Calculate monthly traffic
            monthly_traffic = filtered_df.groupby('month')['traffic_volume'].mean().reset_index()
            
            # Map month numbers to month names
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            
            monthly_traffic['month_name'] = monthly_traffic['month'].map(month_names)
            
            # Sort by month
            monthly_traffic = monthly_traffic.sort_values('month')
            
            # Plot monthly traffic
            fig = px.line(
                monthly_traffic,
                x='month_name',
                y='traffic_volume',
                markers=True,
                title=None,
                labels={'month_name': 'Month', 'traffic_volume': 'Average Traffic Volume'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly statistics
            st.markdown("<h3>Monthly Traffic Statistics</h3>", unsafe_allow_html=True)
            
            # Calculate additional monthly metrics
            monthly_metrics = filtered_df.groupby('month').agg({
                'traffic_volume': 'mean',
                'avg_vehicle_speed': 'mean',
                'accident_reported': 'sum'
            }).reset_index()
            
            monthly_metrics['month_name'] = monthly_metrics['month'].map(month_names)
            monthly_metrics = monthly_metrics.sort_values('month')
            
            # Format for display
            monthly_display = monthly_metrics.copy()
            monthly_display['traffic_volume'] = monthly_display['traffic_volume'].round(1)
            monthly_display['avg_vehicle_speed'] = monthly_display['avg_vehicle_speed'].round(1)
            
            monthly_display.rename(columns={
                'month_name': 'Month',
                'traffic_volume': 'Avg Traffic Volume',
                'avg_vehicle_speed': 'Avg Speed (mph)',
                'accident_reported': 'Accidents'
            }, inplace=True)
            
            monthly_display = monthly_display.drop(columns=['month'])
            
            st.dataframe(monthly_display, use_container_width=True)
            
            # Monthly comparison of vehicle types
            st.markdown("<h3>Vehicle Distribution by Month</h3>", unsafe_allow_html=True)
            
            monthly_vehicles = filtered_df.groupby('month').agg({
                'vehicle_count_cars': 'mean',
                'vehicle_count_trucks': 'mean',
                'vehicle_count_bikes': 'mean'
            }).reset_index()
            
            monthly_vehicles['month_name'] = monthly_vehicles['month'].map(month_names)
            monthly_vehicles = monthly_vehicles.sort_values('month')
            
            vehicles_melted = pd.melt(
                monthly_vehicles,
                id_vars=['month', 'month_name'],
                value_vars=['vehicle_count_cars', 'vehicle_count_trucks', 'vehicle_count_bikes'],
                var_name='Vehicle Type',
                value_name='Average Count'
            )
            
            vehicles_melted['Vehicle Type'] = vehicles_melted['Vehicle Type'].map({
                'vehicle_count_cars': 'Cars',
                'vehicle_count_trucks': 'Trucks',
                'vehicle_count_bikes': 'Bikes'
            })
            
            fig = px.line(
                vehicles_melted,
                x='month_name',
                y='Average Count',
                color='Vehicle Type',
                markers=True,
                title=None,
                labels={'month_name': 'Month', 'Average Count': 'Average Vehicle Count'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Monthly Insights:</h3>", unsafe_allow_html=True)
            
            # Find peak and lowest traffic months
            peak_month = monthly_metrics.loc[monthly_metrics['traffic_volume'].idxmax()]
            lowest_month = monthly_metrics.loc[monthly_metrics['traffic_volume'].idxmin()]
            
            # Find month with most accidents
            accident_month = monthly_metrics.loc[monthly_metrics['accident_reported'].idxmax()]
            
            # Month with most bikes
            bike_month_data = filtered_df.groupby('month')['vehicle_count_bikes'].mean()
            bike_month = bike_month_data.idxmax()
            bike_month_name = month_names[bike_month]
            
            st.markdown(f"• Peak traffic occurs in **{peak_month['month_name']}** with an average volume of **{peak_month['traffic_volume']:.1f}**", unsafe_allow_html=True)
            st.markdown(f"• Lowest traffic occurs in **{lowest_month['month_name']}** with an average volume of **{lowest_month['traffic_volume']:.1f}**", unsafe_allow_html=True)
            st.markdown(f"• Most accidents reported in **{accident_month['month_name']}** ({int(accident_month['accident_reported'])} incidents)", unsafe_allow_html=True)
            st.markdown(f"• Bicycle traffic peaks in **{bike_month_name}**", unsafe_allow_html=True)
            
            # Calculate seasonal variations
            if len(monthly_metrics) >= 3:
                winter_months = [12, 1, 2]
                summer_months = [6, 7, 8]
                
                winter_traffic = monthly_metrics[monthly_metrics['month'].isin(winter_months)]['traffic_volume'].mean()
                summer_traffic = monthly_metrics[monthly_metrics['month'].isin(summer_months)]['traffic_volume'].mean()
                
                if not np.isnan(winter_traffic) and not np.isnan(summer_traffic) and summer_traffic > 0:
                    seasonal_diff = ((winter_traffic - summer_traffic) / summer_traffic) * 100
                    st.markdown(f"• Winter traffic is **{abs(seasonal_diff):.1f}%** {'higher' if seasonal_diff > 0 else 'lower'} than summer traffic", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif pattern_type == "Signal Status Impact":
            st.markdown("<h2 class='sub-header'>Traffic Signal Status Analysis</h2>", unsafe_allow_html=True)
            
            # Traffic volume by signal status
            signal_traffic = filtered_df.groupby('signal_status').agg({
                'traffic_volume': 'mean',
                'avg_vehicle_speed': 'mean'
            }).reset_index()
            
            # Sort by traffic volume
            signal_traffic = signal_traffic.sort_values('traffic_volume', ascending=False)
            
            # Plot traffic volume by signal status
            fig = px.bar(
                signal_traffic,
                x='signal_status',
                y='traffic_volume',
                color='signal_status',
                title=None,
                labels={'signal_status': 'Signal Status', 'traffic_volume': 'Average Traffic Volume'}
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Speed by signal status
            st.markdown("<h3>Vehicle Speed by Signal Status</h3>", unsafe_allow_html=True)
            
            fig = px.bar(
                signal_traffic,
                x='signal_status',
                y='avg_vehicle_speed',
                color='signal_status',
                title=None,
                labels={'signal_status': 'Signal Status', 'avg_vehicle_speed': 'Average Vehicle Speed (mph)'}
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal status distribution
            st.markdown("<h3>Signal Status Distribution</h3>", unsafe_allow_html=True)
            
            signal_dist = filtered_df['signal_status'].value_counts().reset_index()
            signal_dist.columns = ['Signal Status', 'Count']
            signal_dist['Percentage'] = (signal_dist['Count'] / signal_dist['Count'].sum()) * 100
            
            fig = px.pie(
                signal_dist,
                values='Percentage',
                names='Signal Status',
                title=None,
                hole=0.4,
                color='Signal Status',
                color_discrete_map={'Red': 'red', 'Yellow': 'gold', 'Green': 'green'}
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal status by hour
            st.markdown("<h3>Signal Status Distribution by Hour</h3>", unsafe_allow_html=True)
            
            hourly_signals = filtered_df.groupby(['hour', 'signal_status']).size().reset_index(name='count')
            hourly_signals_pivot = hourly_signals.pivot(index='hour', columns='signal_status', values='count').fillna(0)
            
            # Calculate percentage
            hourly_signals_pivot_pct = hourly_signals_pivot.div(hourly_signals_pivot.sum(axis=1), axis=0) * 100
            hourly_signals_pct_melted = hourly_signals_pivot_pct.reset_index().melt(
                id_vars=['hour'],
                value_name='Percentage',
                var_name='Signal Status'
            )
            
            fig = px.area(
                hourly_signals_pct_melted,
                x='hour',
                y='Percentage',
                color='Signal Status',
                title=None,
                labels={'hour': 'Hour of Day', 'Percentage': 'Percentage (%)', 'Signal Status': 'Signal Status'},
                color_discrete_map={'Red': 'red', 'Yellow': 'gold', 'Green': 'green'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=20, b=50, l=50, r=50),
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Signal Insights:</h3>", unsafe_allow_html=True)
            
            # Find signal with highest/lowest traffic
            highest_traffic_signal = signal_traffic.iloc[0]['signal_status']
            highest_traffic_volume = signal_traffic.iloc[0]['traffic_volume']
            
            lowest_traffic_signal = signal_traffic.iloc[-1]['signal_status']
            lowest_traffic_volume = signal_traffic.iloc[-1]['traffic_volume']
            
            # Find signal with highest/lowest speed
            speed_signal = signal_traffic.sort_values('avg_vehicle_speed', ascending=False)
            highest_speed_signal = speed_signal.iloc[0]['signal_status']
            highest_speed = speed_signal.iloc[0]['avg_vehicle_speed']
            
            # Find most common signal status
            most_common_signal = signal_dist.iloc[0]['Signal Status']
            most_common_signal_pct = signal_dist.iloc[0]['Percentage']
            
            st.markdown(f"• Highest traffic volume occurs during **{highest_traffic_signal}** signal status ({highest_traffic_volume:.1f})", unsafe_allow_html=True)
            st.markdown(f"• Lowest traffic volume occurs during **{lowest_traffic_signal}** signal status ({lowest_traffic_volume:.1f})", unsafe_allow_html=True)
            st.markdown(f"• Highest vehicle speed occurs during **{highest_speed_signal}** signal status ({highest_speed:.1f} mph)", unsafe_allow_html=True)
            st.markdown(f"• Most common signal status is **{most_common_signal}** ({most_common_signal_pct:.1f}%)", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif pattern_type == "Accident Analysis":
            st.markdown("<h2 class='sub-header'>Traffic Accident Analysis</h2>", unsafe_allow_html=True)
            
            # Filter for accident data
            accident_data = filtered_df[filtered_df['accident_reported'] == 1]
            non_accident_data = filtered_df[filtered_df['accident_reported'] == 0]
            
            # Calculate accident rate
            total_records = len(filtered_df)
            accident_count = len(accident_data)
            accident_rate = (accident_count / total_records) * 100 if total_records > 0 else 0
            
            # Display accident rate
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Overall Accident Rate</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{accident_rate:.2f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{accident_count} accidents out of {total_records} records</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Accidents by hour
            st.markdown("<h3>Accidents by Hour of Day</h3>", unsafe_allow_html=True)
            
            hourly_accidents = accident_data.groupby('hour').size().reset_index(name='accidents')
            hourly_total = filtered_df.groupby('hour').size().reset_index(name='total')
            
            hourly_merged = pd.merge(hourly_accidents, hourly_total, on='hour')
            hourly_merged['accident_rate'] = (hourly_merged['accidents'] / hourly_merged['total']) * 100
            
            fig = px.bar(
                hourly_merged,
                x='hour',
                y='accident_rate',
                title=None,
                labels={'hour': 'Hour of Day', 'accident_rate': 'Accident Rate (%)'},
                color='accident_rate',
                color_continuous_scale=px.colors.sequential.Reds
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=50, l=50, r=50),
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Accidents by weather condition
            st.markdown("<h3>Accidents by Weather Condition</h3>", unsafe_allow_html=True)
            
            weather_accidents = accident_data.groupby('weather_condition').size().reset_index(name='accidents')
            weather_total = filtered_df.groupby('weather_condition').size().reset_index(name='total')
            
            weather_merged = pd.merge(weather_accidents, weather_total, on='weather_condition')
            weather_merged['accident_rate'] = (weather_merged['accidents'] / weather_merged['total']) * 100
            
            # Sort by accident rate
            weather_merged = weather_merged.sort_values('accident_rate', ascending=False)
            
            fig = px.bar(
                weather_merged,
                x='weather_condition',
                y='accident_rate',
                title=None,
                labels={'weather_condition': 'Weather Condition', 'accident_rate': 'Accident Rate (%)'},
                color='accident_rate',
                color_continuous_scale=px.colors.sequential.Reds
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Accidents by signal status
            st.markdown("<h3>Accidents by Signal Status</h3>", unsafe_allow_html=True)
            
            signal_accidents = accident_data.groupby('signal_status').size().reset_index(name='accidents')
            signal_total = filtered_df.groupby('signal_status').size().reset_index(name='total')
            
            signal_merged = pd.merge(signal_accidents, signal_total, on='signal_status')
            signal_merged['accident_rate'] = (signal_merged['accidents'] / signal_merged['total']) * 100
            
            # Sort by accident rate
            signal_merged = signal_merged.sort_values('accident_rate', ascending=False)
            
            fig = px.bar(
                signal_merged,
                x='signal_status',
                y='accident_rate',
                title=None,
                labels={'signal_status': 'Signal Status', 'accident_rate': 'Accident Rate (%)'},
                color='signal_status',
                color_discrete_map={'Red': 'red', 'Yellow': 'gold', 'Green': 'green'}
            )
            
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare speed distribution in accident vs non-accident cases
            st.markdown("<h3>Speed Distribution: Accident vs Non-Accident Cases</h3>", unsafe_allow_html=True)
            
            accident_data['Case Type'] = 'Accident'
            non_accident_sample = non_accident_data.sample(min(len(accident_data) * 2, len(non_accident_data)))
            non_accident_sample['Case Type'] = 'No Accident'
            
            speed_comparison = pd.concat([accident_data, non_accident_sample])
            
            fig = px.histogram(
                speed_comparison,
                x='avg_vehicle_speed',
                color='Case Type',
                marginal='box',
                nbins=30,
                title=None,
                labels={'avg_vehicle_speed': 'Average Vehicle Speed (mph)', 'count': 'Count', 'Case Type': 'Case Type'},
                color_discrete_map={'Accident': 'red', 'No Accident': 'blue'}
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Accident Insights:</h3>", unsafe_allow_html=True)
            
            # Most dangerous hour
            dangerous_hour = hourly_merged.loc[hourly_merged['accident_rate'].idxmax()]
            
            # Most dangerous weather
            dangerous_weather = weather_merged.iloc[0]
            
            # Most dangerous signal status
            dangerous_signal = signal_merged.iloc[0]
            
            # Speed comparison
            accident_avg_speed = accident_data['avg_vehicle_speed'].mean()
            non_accident_avg_speed = non_accident_data['avg_vehicle_speed'].mean()
            speed_diff_pct = ((accident_avg_speed - non_accident_avg_speed) / non_accident_avg_speed) * 100
            
            st.markdown(f"• Most dangerous time: **{int(dangerous_hour['hour'])}:00** hours ({dangerous_hour['accident_rate']:.2f}% accident rate)", unsafe_allow_html=True)
            st.markdown(f"• Highest accident rate in **{dangerous_weather['weather_condition']}** weather ({dangerous_weather['accident_rate']:.2f}%)", unsafe_allow_html=True)
            st.markdown(f"• Highest accident rate during **{dangerous_signal['signal_status']}** signal ({dangerous_signal['accident_rate']:.2f}%)", unsafe_allow_html=True)
            st.markdown(f"• Average speed in accident cases is **{abs(speed_diff_pct):.1f}%** {'higher' if speed_diff_pct > 0 else 'lower'} than non-accident cases", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif page == "Real-time Monitor":
        st.markdown("<h1 class='main-header'>Real-time Traffic Monitor</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <p style='font-size: 1.1rem;'>Monitor live traffic conditions and receive alerts for congestion or incidents.</p>
        """, unsafe_allow_html=True)
        
        # Location selector
        monitor_location = st.selectbox(
            "Select Location to Monitor",
            options=sorted(df['location_id'].unique()),
            index=0
        )
        
        # Filter data for selected location
        location_data = filtered_df[filtered_df['location_id'] == monitor_location].copy()
        
        if len(location_data) > 0:
            # Use the most recent data points
            recent_data = location_data.sort_values('timestamp', ascending=False).head(60)
            recent_data = recent_data.sort_values('timestamp')
            
            # Current metrics
            current_data = recent_data.iloc[-1] if len(recent_data) > 0 else None
            
            if current_data is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Current Traffic Volume</p>", unsafe_allow_html=True)
                    
                    # Determine traffic status
                    avg_volume = location_data['traffic_volume'].mean()
                    volume_ratio = current_data['traffic_volume'] / avg_volume if avg_volume > 0 else 1
                    
                    volume_status = "Normal"
                    if volume_ratio > 1.5:
                        volume_status = "Heavy"
                    elif volume_ratio < 0.5:
                        volume_status = "Light"
                    
                    volume_color = "green"
                    if volume_status == "Heavy":
                        volume_color = "red"
                    elif volume_status == "Light":
                        volume_color = "blue"
                    
                    st.markdown(f"<p class='metric-value'>{current_data['traffic_volume']:.0f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: {volume_color};'><b>{volume_status}</b> Traffic</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Current Speed</p>", unsafe_allow_html=True)
                    
                    # Determine speed status
                    avg_speed = location_data['avg_vehicle_speed'].mean()
                    speed_ratio = current_data['avg_vehicle_speed'] / avg_speed if avg_speed > 0 else 1
                    
                    speed_status = "Normal"
                    if speed_ratio < 0.7:
                        speed_status = "Slow"
                    elif speed_ratio > 1.3:
                        speed_status = "Fast"
                    
                    speed_color = "green"
                    if speed_status == "Slow":
                        speed_color = "red"
                    elif speed_status == "Fast":
                        speed_color = "orange"
                    
                    st.markdown(f"<p class='metric-value'>{current_data['avg_vehicle_speed']:.1f} mph</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: {speed_color};'><b>{speed_status}</b> Traffic Flow</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Current Conditions</p>", unsafe_allow_html=True)
                    
                    weather_icon = "☀️"
                    if current_data['weather_condition'] == "Cloudy":
                        weather_icon = "☁️"
                    elif current_data['weather_condition'] == "Rainy":
                        weather_icon = "🌧️"
                    elif current_data['weather_condition'] == "Foggy":
                        weather_icon = "🌫️"
                    elif current_data['weather_condition'] == "Windy":
                        weather_icon = "💨"
                    
                    signal_icon = "🔴"
                    if current_data['signal_status'] == "Green":
                        signal_icon = "🟢"
                    elif current_data['signal_status'] == "Yellow":
                        signal_icon = "🟡"
                    
                    st.markdown(f"<p class='metric-value'>{weather_icon} {signal_icon}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>{current_data['weather_condition']} | {current_data['signal_status']} Signal</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display time last updated
                current_time = current_data['timestamp']
                st.markdown(f"<p style='text-align: center; color: #9e9e9e;'>Last updated: {current_time}</p>", unsafe_allow_html=True)
                
                # Traffic history chart
                st.markdown("<h2 class='sub-header'>Recent Traffic History</h2>", unsafe_allow_html=True)
                
                fig = px.line(
                    recent_data,
                    x='timestamp',
                    y='traffic_volume',
                    markers=True,
                    title=None,
                    labels={'timestamp': 'Time', 'traffic_volume': 'Traffic Volume'}
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Speed history chart
                st.markdown("<h2 class='sub-header'>Recent Speed History</h2>", unsafe_allow_html=True)
                
                fig = px.line(
                    recent_data,
                    x='timestamp',
                    y='avg_vehicle_speed',
                    markers=True,
                    title=None,
                    labels={'timestamp': 'Time', 'avg_vehicle_speed': 'Average Speed (mph)'}
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Real-time alerts
                st.markdown("<h2 class='sub-header'>Traffic Alerts</h2>", unsafe_allow_html=True)
                
                alerts = []
                
                # Check for congestion
                if volume_ratio > 1.5:
                    alerts.append({
                        'title': 'Heavy Traffic Alert',
                        'description': f'Traffic volume is {volume_ratio:.1f}x higher than average at this location.',
                        'severity': 'High',
                        'color': 'red'
                    })
                
                # Check for slow speeds
                if speed_ratio < 0.7:
                    alerts.append({
                        'title': 'Slow Traffic Alert',
                        'description': f'Current traffic speed is {(1-speed_ratio)*100:.0f}% below average at this location.',
                        'severity': 'Medium',
                        'color': 'orange'
                    })
                
                # Check for accidents
                if current_data['accident_reported'] == 1:
                    alerts.append({
                        'title': 'Accident Reported',
                        'description': 'An accident has been reported at this location. Expect delays.',
                        'severity': 'High',
                        'color': 'red'
                    })
                
                # Check for bad weather
                if current_data['weather_condition'] in ['Rainy', 'Foggy']:
                    alerts.append({
                        'title': f'{current_data["weather_condition"]} Weather Alert',
                        'description': f'{current_data["weather_condition"]} conditions may affect visibility and road safety.',
                        'severity': 'Medium',
                        'color': 'orange'
                    })
                
                # Display alerts
                if alerts:
                    for alert in alerts:
                        st.markdown(f"""
                        <div style="padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; background-color: rgba({255 if alert['color'] == 'red' else 255 if alert['color'] == 'orange' else 0}, {165 if alert['color'] == 'orange' else 0}, 0, 0.1); border-left: 5px solid {alert['color']};">
                            <h3 style="margin-top: 0; color: {alert['color']};">{alert['title']}</h3>
                            <p>{alert['description']}</p>
                            <p style="margin-bottom: 0;"><strong>Severity:</strong> {alert['severity']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("No active alerts at this location. Traffic is flowing normally.")
                
                # Automatic refresh button
                st.markdown("<h3>Real-Time Updates</h3>", unsafe_allow_html=True)
                
                if st.button("↻ Refresh Data"):
                    st.rerun()
                
                st.markdown("<p style='color: #9e9e9e;'>Click the refresh button to get the latest traffic data.</p>", unsafe_allow_html=True)
                
                # Simulated real-time update
                # In a real app, this would use actual real-time data
                with st.expander("⚙️ Simulate Real-Time Updates"):
                    interval = st.slider("Update Interval (seconds)", min_value=5, max_value=60, value=30)
                    
                    if st.button("Start Auto-Refresh"):
                        st.session_state.auto_refresh = True
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(interval):
                            progress_bar.progress((i + 1) / interval)
                            status_text.text(f"Next update in {interval - i} seconds...")
                            time.sleep(1)
                        
                        st.rerun()
            else:
                st.warning("No recent data available for this location.")
        else:
            st.warning("No data available for the selected location.")
    
    # Footer
    st.markdown("<div class='footer'>Traffic Flow Analysis & Prediction Dashboard | Powered by Streamlit</div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()