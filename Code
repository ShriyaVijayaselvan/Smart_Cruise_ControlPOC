import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
import skfuzzy as fuzz
import requests

# Function to get simulated sensor data from Kaggle dataset
def get_sensor_data():
    # Simulating data retrieval from a Kaggle dataset
    try:
        data = pd.read_csv('simulated_sensor_data.csv')
        return data
    except FileNotFoundError:
        print("Error: simulated_sensor_data.csv not found. Please create the csv or comment out the function")
        return None
    except Exception as e:
        print(f"An error occurred while reading sensor data: {e}")
        return None

# Function to get weather data from OpenWeatherMap API
def get_weather_data(api_key, location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while processing weather data: {e}")
        return None

# Generate simulated data for the POC
def get_simulated_data():
    time = np.arange(0, 100, 1)
    speed = np.sin(time / 10) * 20 + 50  # Simulated speed data
    fuel_efficiency = np.random.uniform(0.8, 1.2, size=time.shape)  # Simulated fuel efficiency data
    terrain = np.sin(time / 15) * 5  # Simulated terrain data
    return time, speed, fuel_efficiency, terrain

# Define RNN model for speed prediction
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Define CNN model for terrain analysis
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Define fuzzy logic controller for fuel optimization
def fuzzy_fuel_optimization(speed, terrain):
    speed = np.clip(speed, 0, 100)
    terrain = np.clip(terrain, -10, 10)
    speed_range = np.arange(0, 101, 1)
    terrain_range = np.arange(-10, 11, 1)
    fuel_range = np.arange(0.5, 1.6, 0.1)

    speed_fuzz = fuzz.trimf(speed_range, [0, 50, 100])
    terrain_fuzz = fuzz.trimf(terrain_range, [-10, 0, 10])
    fuel_fuzz = fuzz.trimf(fuel_range, [0.5, 1.0, 1.5])

    speed_level = fuzz.interp_membership(speed_range, speed_fuzz, speed)
    terrain_level = fuzz.interp_membership(terrain_range, terrain_fuzz, terrain)

    rule1 = np.fmin(speed_level, terrain_level)
    rule2 = np.fmin(speed_level, np.fmax(terrain_level, fuel_fuzz))

    aggregated = np.fmax(rule1, rule2)
    fuel_efficiency = fuzz.defuzz(fuel_range, aggregated, 'centroid')
    
    return fuel_efficiency

# Emergency braking system
def emergency_braking(speed, threshold=30):
    if speed > threshold:
        return True
    return False

# Main function to integrate all components
def main():
    # Load sensor data and weather data
    sensor_data = get_sensor_data()
    weather_data = get_weather_data('your_openweather_api_key', 'your_location')

    # Simulate data
    time, speed, fuel_efficiency, terrain = get_simulated_data()

    # Prepare data for RNN
    X_speed = speed[:-1].reshape((len(speed) - 1, 1, 1))
    y_speed = speed[1:]

    # Create and train RNN model
    rnn_model = create_rnn_model((1, 1))
    rnn_model.fit(X_speed, y_speed, epochs=50, verbose=1)

    # Predict speed using RNN
    speed_predictions = rnn_model.predict(X_speed)

    # Create and train CNN model for terrain analysis
    X_terrain = terrain[:-1].reshape((len(terrain) - 1, 1, 1))
    y_terrain = terrain[1:]
    cnn_model = create_cnn_model((1, 1))
    cnn_model.fit(X_terrain, y_terrain, epochs=50, verbose=1)

    # Analyze terrain using CNN
    terrain_predictions = cnn_model.predict(X_terrain)

    # Apply fuzzy logic for fuel optimization
    optimized_fuel = []
    for s, t in zip(speed, terrain):
        optimized_fuel.append(fuzzy_fuel_optimization(s, t))

    # Check for emergency braking
    braking_signals = [emergency_braking(s) for s in speed]

    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time[:-1], speed_predictions, label='Predicted Speed')
    plt.plot(time, speed, label='Actual Speed', alpha=0.5)
    plt.legend()
    plt.title('Speed Prediction')

    plt.subplot(3, 1, 2)
    plt.plot(time[:-1], terrain_predictions, label='Predicted Terrain')
    plt.plot(time, terrain, label='Actual Terrain', alpha=0.5)
    plt.legend()
    plt.title('Terrain Analysis')

    plt.subplot(3, 1, 3)
    plt.plot(time, optimized_fuel, label='Optimized Fuel Efficiency')
    plt.legend()
    plt.title('Fuel Optimization')

    plt.tight_layout()
    plt.show()

    # Print emergency braking signals
    for t, brake in zip(time, braking_signals):
        if brake:
            print(f"Emergency braking triggered at time {t}!")

if __name__ == "__main__":
    main()
