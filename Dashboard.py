import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import streamlit as st
import plotly.graph_objects as go
import sklearn
import os

def CurrentPrice(name):
    if category == "Processed Food":
        data = pd.read_csv("averaged/Processed Food/"+name+".csv",index_col=False)
    elif category == "Raw Food":
        data = pd.read_csv("averaged/Raw Food/"+name+".csv",index_col=False)
    data['Percentage Difference'] = data.groupby('state')['price'].pct_change() * 100
    # Filtering only the latest date prices
    df = data.groupby('state').tail(1)
    df.reset_index(inplace=True)
    df.index += 1 


    df = df.rename(columns={'price': 'Price'})
    df = df.rename(columns={'state': 'State'})
    
    df['Price'] = 'RM ' + df['Price'].round(2).astype(str)

    
    
    df['Percentage Difference'] = \
        df['Percentage Difference'].apply(lambda x: f"{'+' if x > 0 else ''}{round(x, 2)}%{'📈' if x > 0 else ''}{'📉' if x < 0 else ''}" if pd.notna(x) else "")
                                    # .apply(lambda x: f"{'+' if x > 0 else ''}{round(x, 2)}%" if pd.notna(x) else "")
    


    st.dataframe(df[['State',  'Price', 'Percentage Difference']],hide_index=True, use_container_width=True,height=602)
    
def SelectBox(category):
    if category == "Processed Food":
        csv_files = [file for file in os.listdir("averaged/Processed Food") if file.endswith(".csv")]

    elif category == "Raw Food":
        csv_files = [file for file in os.listdir("averaged/Raw Food") if file.endswith(".csv")]
    itemls = []
    for file in csv_files:
        itemls.append(file[:-4])

    name = st.selectbox("Item Name",itemls)
    return name
    
# Load the trained GP model
def DisplayGraph(state, name):

    data = pd.read_csv("movavg/" + name + ".csv")
    # data1 = pd.read_csv("C:/Users/Asus/Desktop/FYP/DATA/averaged/" + name + ".csv")
    # Convert date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    
    

    # Filter the data for the chosen state
    filtered_data = data[data['state'] == state]
    # filtered_data1 = data1[data1['state'] == state]

    model_filename = f"savedModel/processed food mmodel({state}){name}.pkl"
    try:
        # Try to load the model
        with open(model_filename, 'rb') as f:
            loaded_model = joblib.load(f)
    except FileNotFoundError:
        # Handle the FileNotFoundError
        st.info(f"The item '{name}' for state '{state}' does not have a saved model. Please try another item or state.")
        return

    

    start_dates = filtered_data['date'].min()
    end_dates = filtered_data['date'].max()+ timedelta(days=1 * 30)
    date_test = pd.date_range(start=start_dates, end=end_dates, freq='D')
    
    date_range = (filtered_data['date'].max()-filtered_data['date'].min()).days
    reference_date = datetime(2023, 1, 1) 

    normalized_date_test = (date_test - reference_date).days / date_range

    X_test = normalized_date_test.values.reshape(-1, 1)

    y_pred_test_loaded, sigma_range_loaded = loaded_model.predict(X_test, return_std=True)

    predicted_price_test_loaded = y_pred_test_loaded * np.max(filtered_data['price'])

    # Calculate the latest actual price
    latest_actual_price = filtered_data['price'].iloc[-2]

    # Calculate the percentage difference for the test data
    percentage_difference_test = ((predicted_price_test_loaded[-1] - latest_actual_price) / latest_actual_price) * 100

    # st.write(percentage_difference_test)

    # Create an interactive plot using Plotly
    fig = go.Figure()

    # Actual Price trace
    fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['price'], mode='markers', name='Actual Price', marker=dict(color='#3498db')))
    # fig.add_trace(go.Scatter(x=filtered_data1['date'], y=filtered_data1['price'], mode='markers', name='Actual Price', marker=dict(color='#00ff00')))
    # Predicted Price trace
    fig.add_trace(go.Scatter(x=date_test, y=predicted_price_test_loaded, mode='lines', name='Predicted Price', line=dict(color='#e74c3c')))

    # Shaded uncertainty area
    fig.add_trace(go.Scatter(
        x=np.concatenate([date_test, date_test[::-1]]),
        y=np.concatenate([predicted_price_test_loaded - 1 * sigma_range_loaded, (predicted_price_test_loaded + 1 * sigma_range_loaded)[::-1]]),
        fill='toself',
        fillcolor='rgba(250,60,60,0.3)',  # Change the color here
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty'
    ))

    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        title=f'Price for {name} in {state}',
        showlegend=True,
    )

    # Display the interactive plot using Streamlit
    st.plotly_chart(fig)
    return float(percentage_difference_test)

unique_states = ['Choose a State...','Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan',
                  'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor', 
                  'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']   

# Streamlit App
col1,col2=st.columns([0.2,1])
with col1:
    st.write("")
    # st.image("dashboard/logo.png",width=100)
with col2:
    st.title("HargaBarangNow")
    


st.write("Welcome to the Website for data and insights on food prices.")
st.caption("Last updated on November 2023")
st.write("")



tab1, tab2 = st.tabs(["Price Trend", "Current Price"])

with tab1:
    st.subheader("This shows the price trends of the chosen food in a state.")
    st.write("")
    col1, col2, col3 = st.columns([0.3,1,0.1])
    with col1:
        # st.subheader("Choose your State and Item")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        chosen_state = st.selectbox('Select State', unique_states)
        category = st.selectbox('Select Food Category', ["Processed Food", "Raw Food"])
        if category == "Processed Food":
            name = st.selectbox('Select Item', ["Choose an item...", "COCA COLA (BOTOL),1.5 liter","COCA COLA (TIN),320ml", "DUTCH LADY UHT COKLAT (KOTAK),200ml","DUTCH LADY UHT FULL CREAM (KOTAK),200ml"
                                        ,"F&N OREN (BOTOL),1.5 liter","F&N OREN (TIN),325 ml","HORLICKS (PAKET),400g","KICAP LEMAK MANIS CAP KIPAS UDANG,345ml"
                                        ,"KICAP MASIN ADABI,340ml","KICAP MANIS ADABI,340ml","KORDIAL SUNQUICK (OREN),840 ml","KRIMER MANIS PEKAT CAP SAJI,500g"
                                        ,"KRIMER SEJAT CAP F&N,390g","MACKAREL CAP AYAM (SOS TOMATO),425g","MAGGI MI SEGERA PERISA KARI,5 X 79g","MARJERIN DAISY,240g"
                                        ,"MENTEGA ANCHOR (SALTED),227g","MENTEGA KACANG HALUS LADY'S CHOIE,340g","MI SEDAP MI GORENG PERISA ASLI,5 X 90g"
                                        ,"MILO (PAKET),1kg","MILO (PAKET),400g"])
            
        elif category == "Raw Food":
            name = st.selectbox('Select Item', ["Choose an item...", "COCA COLA (BOTOL),1.5 liter","COCA COLA (TIN),320ml", "DUTCH LADY UHT COKLAT (KOTAK),200ml","DUTCH LADY UHT FULL CREAM (KOTAK),200ml"
                            ,"F&N OREN (BOTOL),1.5 liter","F&N OREN (TIN),325 ml","HORLICKS (PAKET),400g","KICAP LEMAK MANIS CAP KIPAS UDANG,345ml"
                            ,"KICAP MASIN ADABI,340ml","KICAP MANIS ADABI,340ml"])

        # Display date picker in the second column
    with col2:
        # selected_date = st.date_input("Select a Date", datetime(2023, 1, 1))

        # Display Instructions when No Selection is Made
        dummy_fig = go.Figure()
        dummy_date = pd.to_datetime('2023-01-01')
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

            # Create two columns for layout
            # col1, col2 = st.columns([0.4  ,2])

            # Display graph in the first column
            # with col1:
            percentage_difference=DisplayGraph(chosen_state, name)
    with col1:
        
        if ((chosen_state != 'Choose a State...') and (name != 'Choose an item...')):
            if percentage_difference is not None:

                if percentage_difference > 0:
                    st.subheader(f":red[+{str(round(percentage_difference,2))}%]")
                elif percentage_difference < 0:
                    st.subheader(f":blue[{str(round(percentage_difference,2))}%]")
        
        # Display date picker in the second column
        # with col2:
            # selected_date = st.date_input("Select a Date", datetime(2023, 1, 1))
with tab2:
    st.subheader("Percentage change in price of the chosen item from previous month in every states.")

    col1, col2, col3 = st.columns([0.3,0.3,0.3])
    with col1:
        # st.write("😊📉📈")
        category=st.selectbox("Food Categories",(['Processed Food','Raw Food']))
        
    with col2:
        name = SelectBox(category)
    # with col2:
    st.write("")

    CurrentPrice(name,category)

