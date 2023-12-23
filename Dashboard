import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

# Load the trained GP model
def DisplayGraph(state, name):
    path = "C:/Users/Asus/Desktop/FYP/DATA/movavg/"
    file = name
    format = ".csv"
    data = pd.read_csv(path + file + format)

    # Convert date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    # Define the chosen state for plotting
    chosen_state = state

    # Filter the data for the chosen state
    filtered_data = data[data['state'] == chosen_state]

    model_filename = f"C:/Users/Asus/Desktop/FYP/DATA/savedModel/({chosen_state}){file}.pkl"
    try:
        # Try to load the model
        with open(model_filename, 'rb') as f:
            loaded_model = joblib.load(f)
    except FileNotFoundError:
        # Handle the FileNotFoundError
        st.info(f"The item '{file}' for state '{chosen_state}' does not have a saved model. Please try another item or state.")
        return

    state_mapping = {s: i for i, s in enumerate(set(filtered_data['state']))}

    start_dates = filtered_data['date'].min()
    end_dates = filtered_data['date'].max()+ timedelta(days=1 * 30)
    date_test = pd.date_range(start=start_dates, end=end_dates, freq='D')
    reference_date = datetime(2022, 9, 1)

    normalized_date_test = (date_test - reference_date).days / ((datetime.now() - reference_date).days)

    encoded_chosen_state_range = np.array([state_mapping[chosen_state]] * len(date_test)).reshape(-1, 1)

    X_test = np.concatenate((encoded_chosen_state_range, normalized_date_test.values.reshape(-1, 1)), axis=1)

    y_pred_test_loaded, sigma_range_loaded = loaded_model.predict(X_test, return_std=True)

    predicted_price_test_loaded = y_pred_test_loaded * np.max(filtered_data['price'])

    # Streamlit App

    # Create an interactive plot using Plotly
    fig = go.Figure()

    # Actual Price trace
    fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['price'], mode='markers', name='Actual Price', marker=dict(color='#3498db')))

    # Predicted Price trace
    fig.add_trace(go.Scatter(x=date_test, y=predicted_price_test_loaded, mode='lines', name='Predicted Price', line=dict(color='#e74c3c')))

    # Shaded uncertainty area
    fig.add_trace(go.Scatter(
        x=np.concatenate([date_test, date_test[::-1]]),
        y=np.concatenate([predicted_price_test_loaded - 1.5 * sigma_range_loaded, (predicted_price_test_loaded + 1.5 * sigma_range_loaded)[::-1]]),
        fill='toself',
        fillcolor='rgba(231,76,60,0.2)',  # Change the color here
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty'
    ))

    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        title=f'Price for {file} in {chosen_state}',
        showlegend=True,
    )

    # Display the interactive plot using Streamlit
    st.plotly_chart(fig)

path = "C:/Users/Asus/Desktop/FYP/DATA/movavg/"
file = "COCA COLA (BOTOL),1.5 liter"
format = ".csv"
data = pd.read_csv(path + file + format)
unique_states = ['Choose a State...'] + sorted(data['state'].unique())

# Streamlit App
st.title("")

st.title("HargaBarangNow")
# st.header("HargaBarangNow in Malaysia")
st.write("Welcome to the Website for data and insights on food prices.")

# Sidebar Layout
st.sidebar.title("Choose your State and Item")
chosen_state = st.sidebar.selectbox('Select a State', unique_states)
name = st.sidebar.selectbox('Select an Item', ["Choose an item...", "COCA COLA (BOTOL),1.5 liter", "DUTCH LADY UHT COKLAT (KOTAK),200ml"])

# Display Instructions when No Selection is Made
dummy_fig = go.Figure()
dummy_date = pd.to_datetime('2022-01-01')
dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='markers', name='Actual Price'))
dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='lines', name='Predicted Price', line=dict(color='#e74c3c')))
dummy_fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price'),
    title='Select a state and an item to view the graph',
    showlegend=True,
)

if ((chosen_state == 'Choose a State...') or (name == 'Choose an item...')):
    # st.info("Please select a state and an item to view the graph.")
    st.plotly_chart(dummy_fig)
else:
    # Call the function to display the graph when a state is chosen
    # st.info(f"Price for {name} in {chosen_state}")
    DisplayGraph(chosen_state, name)

# Footer
# st.sidebar.markdown("Developed by Your Name")

