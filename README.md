# 🚦 Traffic Prediction Analysis Dashboard

This is a Streamlit-powered interactive dashboard for analyzing and visualizing traffic prediction data using big data techniques and Python data science libraries.


## Project Structure

Dashboard_big_data
├── app.py # Main Streamlit app file
├── requirements.txt # Python dependencies
└── README.md # Project documentation
└── smart_traffic_management_dataset.csv # Dataset from kaggle
## Data Description

### Dataset: `Smart Traffic Management`

This dataset records traffic and environmental conditions across multiple locations in a city. The data is suitable for traffic pattern analysis, congestion prediction, and anomaly detection (e.g., accident impact).

| Column Name             | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `timestamp`             | Date and time of the recorded traffic data                                  |
| `location_id`           | Unique identifier for the traffic sensor location                           |
| `traffic_volume`        | Total number of vehicles recorded in the timestamp window                   |
| `avg_vehicle_speed`     | Average speed of vehicles (in km/h) at that time and location               |
| `vehicle_count_cars`    | Number of cars detected                                                     |
| `vehicle_count_trucks`  | Number of trucks detected                                                   |
| `vehicle_count_bikes`   | Number of bikes or two-wheelers detected                                    |
| `weather_condition`     | Weather at the time (e.g., Clear, Rainy, Foggy)                             |
| `temperature`           | Temperature in Celsius                                                      |
| `humidity`              | Humidity percentage                                                         |
| `accident_reported`     | Boolean or categorical (e.g., Yes/No or 1/0) indicating accident occurrence |
| `signal_status`         | Status of traffic signal (e.g., Green, Red, Amber)                          |

**Source**: Kaggle - *Smart Traffic Management Dataset*


## Features

- Interactive visual analytics for traffic data
- Visualize traffic volume and speed over time by location
- Compare vehicle types and conditions
- Weather vs traffic correlation analysis
- Plotly, Seaborn, and Matplotlib-based plots


## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Soni-Tisha/Traffic-Predection-Analysis-Dashboard.git
   cd Traffic-Predection-Analysis-Dashboard

2. **Install dependencies**
   pip install -r requirements.txt
3. **Run the Streamlit app**
   streamlit run Dashboard_big_data/app.py
