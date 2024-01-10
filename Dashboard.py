import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import streamlit as st
import plotly.graph_objects as go
import sklearn
import os
import re


# def MovingAverage(state,name,category):
#     if category == "Processed Food":
#         data = pd.read_csv(f"data/processed food/{name}.csv")
#     elif category == "Raw Food":
#         data = pd.read_csv(f"data/raw food/{name}.csv")

#     data = data[data['state'] == state]
#     # Convert 'date' to datetime
#     data['date'] = pd.to_datetime(data['date'])


#     # Sort DataFrame by date
#     data.sort_values(by='date', inplace=True)

#     # Calculate 3-month moving average
#     data['moving_average'] = data['price'].rolling(window=3).mean()

#     fig = go.Figure()

#     # Original Prices trace
#     fig.add_trace(go.Scatter(x=data['date'], y=data['price'], mode='lines+markers', name='Original Prices', marker=dict(color='blue')))

#     # Moving Average trace
#     fig.add_trace(go.Scatter(x=data['date'], y=data['moving_average'], mode='lines+markers', name='3-Month Moving Average', marker=dict(color='orange')))

#     # Update layout
#     fig.update_layout(
#         title='Original Prices vs 3-Month Moving Average',
#         xaxis_title='Date',
#         yaxis_title='Price',
#         legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)


def CurrentPrice(name,category):
    if category == "Processed Food":
        data = pd.read_csv("data/processed food/"+name+".csv",index_col=False)
    elif category == "Raw Food":
        data = pd.read_csv("data/raw food/"+name+".csv",index_col=False)
    data['Percentage Difference (from OCT 2023)'] = data.groupby('state')['price'].pct_change() * 100
    # Filtering only the latest date prices
    df = data.groupby('state').tail(1)

    df = df.rename(columns={'price': 'Price (NOV 2023)'})
    df = df.rename(columns={'state': 'State'})
    high=df.nlargest(1,["Price (NOV 2023)"])
    low=df.nsmallest(1,["Price (NOV 2023)"]).reset_index()
    
    df['Price (NOV 2023)'] = 'RM ' + df['Price (NOV 2023)'].round(2).astype(str)

        
    df['Percentage Difference (from OCT 2023)'] = \
        df['Percentage Difference (from OCT 2023)'].apply(lambda x: f"{'+' if x > 0 else ''}{round(x, 2)}%{'📈' if x > 0 else ''}{'📉' if x < 0 else ''}" if pd.notna(x) else "")
                                    # .apply(lambda x: f"{'+' if x > 0 else ''}{round(x, 2)}%" if pd.notna(x) else "")
    
    st.write("Percentage difference of price between October 2023 and November 2023 ")
    st.dataframe(df[['State',  'Price (NOV 2023)', 'Percentage Difference (from OCT 2023)']],hide_index=True, use_container_width=True,height=600)

    
    state_low=low['State'].iloc[0]
    price_low=low['Price (NOV 2023)'].iloc[0]

    state_high=high['State'].iloc[0]
    price_high=high['Price (NOV 2023)'].iloc[0]

    st.write(f'Highest Price : RM **{round(price_high,2)}** in **{state_high}**')
    st.write(f'Lowest Price : RM **{round(price_low,2)}** in **{state_low}**')
    
def SelectBox(category):
    if category == "Processed Food":
        csv_files = [file for file in os.listdir("data/processed food") if file.endswith(".csv")]

    elif category == "Raw Food":
        csv_files = [file for file in os.listdir("data/raw food") if file.endswith(".csv")]
    itemls = []
    for file in csv_files:
        itemls.append(file[:-4])

    name = st.selectbox("Item Name",itemls)
    return name
    
# Load the trained GP model
def DisplayGraphProcessed(state, name):

    data = pd.read_csv("data/processed food/" + name + ".csv")
    model_filename = f"model/processed food/({state}){name}.pkl"

    
    # data1 = pd.read_csv("C:/Users/Asus/Desktop/FYP/DATA/averaged/" + name + ".csv")
    # Convert date column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    
    

    # Filter the data for the chosen state
    filtered_data = data[data['state'] == state]
    # filtered_data1 = data1[data1['state'] == state]

    
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

    # Calculate the percentage difference for the test data
    percentage_difference_test = ((predicted_price_test_loaded[-1] - predicted_price_test_loaded[-31]) / predicted_price_test_loaded[-31]) * 100

    # st.write(percentage_difference_test)
    filtered_data['moving_average'] = filtered_data['price'].rolling(window=3).mean()
    # Create an interactive plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['moving_average'], mode='lines+markers', name='3-Month Moving Average', marker=dict(color='orange')))
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
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        template='plotly_white'
    )

    # Display the interactive plot using Streamlit
    st.plotly_chart(fig)
    return float(percentage_difference_test)

def DisplayGraphRaw(state, name):
    path = "data/raw food/"
    file = name
    my_item = ['AYAM BERSIH - STANDARD','BAWANG BESAR IMPORT INDIA','BAWANG KECIL MERAH ROSE IMPORT INDIA','BAWANG PERAI LEEK IMPORT','BAWANG PUTIH IMPORT CHINA','BAYAM HIJAU'
               ,'BAYAM MERAH','BETIK BIASA','CILI AKAR HIJAU','CILI HIJAU','CILI KERING KERINTING BERTANGKAI OR TIDAK BERTANGKAI','DADA AYAM CHICKEN KEEL 1KG',
               'DAGING LEMBU TEMPATAN BAHAGIAN DAGING PAHA KECUALI BATANG PINANG - TENDERLOIN','DRAGON FRUIT MERAH','EPAL HIJAU GRANNY SMITH SAIZ M','IKAN BILIS GRED B KOPEK',
               'IKAN CENCARU ANTARA 4 HINGGA 6 EKOR SEKILOGRAM','IKAN GELAMA ANTARA 5 HINGGA 10 EKOR SEKILOGRAM','IKAN JENAHAK','IKAN JENAHAK KEPINGAN','IKAN KEMBUNG ANTARA 8 HINGGA 12 EKOR SEKILOGRAM',
               'IKAN KEMBUNG KECIL OR PELALING','IKAN KERISI ANTARA 5 HINGGA 10 EKOR SEKILOGRAM','IKAN MABUNG ANTARA 6 HINGGA 10 EKOR SEKILOGRAM','IKAN PARI KEPINGAN',
               'IKAN SELAR PELATA','IKAN SELAYANG OR SARDIN','IKAN SIAKAP ANTARA 2 HINGGA 4 EKOR SEKILOGRAM','IKAN TENGGIRI BATANG KEPINGAN','IKAN TILAPIA MERAH ANTARA 2 HINGGA 5 EKOR SEKILOGRAM',
               'JAMBU BATU BERBIJI','JAMBU BATU TANPA BIJI','KACANG BENDI','KACANG BOTOL','KACANG BUNCIS','KACANG HIJAU IMPORT','KACANG MERAH IMPORT','KACANG PANJANG','KACANG SOYA IMPORT','KACANG TANAH IMPORT',
               'KANGKUNG','KELAPA BIJI','KELAPA PARUT','KEPAK AYAM CHICKEN WING','KEPALA IKAN JENAHAK','KERANG SAIZ SEDERHANA','KETAM RENJONG atau BUNGAANTARA 5 HINGGA 8 EKOR SEKILOGRAM','KUBIS BULAT TEMPATAN',
               'KUBIS BULAT IMPORT BEIJING','KUBIS BULAT IMPORT CHINA','KUBIS BUNGA CAULIFLOWER','KUBIS PANJANG CHINA - BESAR','KUNYIT HIDUP','LADA BENGGALA HIJAU CAPSICUM','LADA BENGGALA KUNING CAPSICUM',
               'LADA BENGGALA MERAH CAPSICUM','LIMAU KASTURI','LIMAU NIPIS','LOBAK MERAH','NENAS BIASA','PAHA AYAM CHICKEN DRUMSTICK','PISANG BERANGAN','PISANG EMAS','SADERI','SANTAN KELAPA SEGAR PEKAT',
               'SOTONG Lebih atau samadengan 6 EKOR SEKILOGRAM','TAUGE KACANG HIJAU','TAUHU JENIS KERAS','TELUR AYAM GRED A','TELUR AYAM GRED B','TELUR AYAM GRED C','TELUR AYAM KAMPUNG','TELUR MASIN 1 biji',
               'TELUR MASIN 4 biji','TEMBIKAI MERAH TANPA BIJI','TEMBIKAI SUSU','TEMPE BUNGKUSAN PLASTIK','TERUNG PANJANG','THIGH AYAM','UBI KENTANG IMPORT CHINA','UDANG HARIMAU ANTARA 20 HINGGA 30 EKOR SEKILOGRAM',
               'UDANG KERING','UDANG PUTIH OR VANNAMEI TERNAK ANTARA 41 HINGGA 60 EKOR SEKILOGRAM','WHOLE LEG AYAM'] #bawang besar import (india ) kena buang bracket kalau nk guna gp
    format = ".csv"
    data = pd.read_csv(path + file + format)
    

    # Convert date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m')

    # Define the chosen state for plotting
    chosen_state = state

    # Filter the data for the chosen state
    filtered_data = data[data['state'] == chosen_state]

    pattern = re.compile(file)

    # Check if any string in the list matches the pattern
    matching_items = [item for item in my_item if pattern.match(item) ]
    if matching_items:
        print(matching_items)
        model_filename = f"model/raw food/{file}/GPR({chosen_state}){file}.pkl"
        print (file + "gp")
        try:
                # Try to load the model
            with open(model_filename, 'rb') as f:
                loaded_model = joblib.load(f)
        except FileNotFoundError:
                # Handle the FileNotFoundError
            st.info(f"The item '{file}' for state '{chosen_state}' does not have a saved model. Please try another item or state.")
            return
            #GPR 
        start = filtered_data['date'].min()
        end = filtered_data['date'].max()+timedelta(1*30)
        range_datetime = (end - start).days
        start_dates = filtered_data['date'].min()
        end_dates = filtered_data['date'].max()+timedelta(1*30)
        date_test = pd.date_range(start=start_dates, end=end_dates, freq='D')
        reference_date = datetime(2022, 10, 1)

        normalized_date_test = (date_test - reference_date).days / range_datetime

        X_test =  normalized_date_test.values.reshape(-1, 1)

        y_pred_test_loaded, sigma_range_loaded = loaded_model.predict(X_test, return_std=True)

        predicted_price_test_loaded = y_pred_test_loaded * np.max(filtered_data['price'])
        percentage_difference_test = ((predicted_price_test_loaded[-1] - predicted_price_test_loaded[-31]) / predicted_price_test_loaded[-31]) * 100
            # Streamlit App

                # Create an interactive plot using Plotly
        filtered_data['moving_average'] = filtered_data['price'].rolling(window=3).mean()
        # Create an interactive plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['moving_average'], mode='lines+markers', name='3-Month Moving Average', marker=dict(color='orange')))

            # Actual Price trace
        fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['price'], mode='markers', name='Actual Price', marker=dict(color='#3498db')))

            # Predicted Price trace
        fig.add_trace(go.Scatter(x=date_test, y=predicted_price_test_loaded, mode='lines', name='Predicted Price', line=dict(color='#e74c3c')))

            # Shaded uncertainty area
        fig.add_trace(go.Scatter(
            x=np.concatenate([date_test, date_test[::-1]]),
            y=np.concatenate([predicted_price_test_loaded - sigma_range_loaded, (predicted_price_test_loaded + sigma_range_loaded)[::-1]]),
            fill='toself',
            fillcolor='rgba(231,76,60,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Uncertainty'
        ))
        fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        title=f'Price for {file} in {chosen_state}',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        template='plotly_white'
        )

            # Display the interactive plot using Streamlit
        st.plotly_chart(fig)
        return percentage_difference_test

    else:
        model_filename = f"model/raw food/{file}/SVR({chosen_state}){file}.pkl"
        print (file + "svr")
        try:
            # Try to load the model
            with open(model_filename, 'rb') as f:
                loaded_model = joblib.load(f)
        except FileNotFoundError:
            # Handle the FileNotFoundError
            st.info(f"The item '{file}' for state '{chosen_state}' does not have a saved model. Please try another item or state.")
            return
        
        start_dates = filtered_data['date'].min()
        end_dates = filtered_data['date'].max()+timedelta(1*30)
        start = filtered_data['date'].min()
        end = filtered_data['date'].max()+timedelta(1*30)
        range_datetime = (end - start).days
        date_test = pd.date_range(start=start_dates, end=end_dates, freq='D')
        reference_date = datetime(2022, 10, 1)
        normalized_date_test_ = (date_test - reference_date).days / range_datetime
    

        X_svr = normalized_date_test_.values.reshape(-1, 1)

        y_pred_test = loaded_model.predict(X_svr)

        # Denormalize the predicted prices
        predicted_price_test = y_pred_test * np.max(filtered_data['price'])
        percentage_difference_test = ((predicted_price_test[-1] - predicted_price_test[-31]) / predicted_price_test[-31]) * 100

         # Create an interactive plot using Plotly
        filtered_data['moving_average'] = filtered_data['price'].rolling(window=3).mean()
        # Create an interactive plot using Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['moving_average'], mode='lines+markers', name='3-Month Moving Average', marker=dict(color='orange')))

        # Actual Price trace
        fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['price'], mode='markers', name='Actual Price', marker=dict(color='#3498db')))

        # Predicted Price trace
        fig.add_trace(go.Scatter(x=date_test, y=predicted_price_test, mode='lines', name='Predicted Price', line=dict(color='#e74c3c')))

        fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        title=f'Price for {file} in {chosen_state}',
        showlegend=True,
        )

        # Display the interactive plot using Streamlit
        st.plotly_chart(fig)
        return percentage_difference_test

if __name__ == "__main__":
    unique_states = ['Choose a State...','Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan',
                      'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor', 
                      'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']   
    
    # Streamlit App
    col1,col2=st.columns([0.2,1])
    with col1:
        st.write("")
        st.image("logo.png",width=100)
    with col2:
        st.title("HargaBarangNow")
        
    
    
    st.write("Welcome to the Website for data and insights on food prices.")
    st.caption("Last updated on November 2023")
    st.write("")
    
    
    
    tab1, tab2 = st.tabs(["Price Trend", "Current Price"])
    
    with tab1:
        st.subheader("This shows the price trends of the chosen food in a state.")
        st.write("")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
    
            chosen_state = st.selectbox('Select State', unique_states,index=None,placeholder="Choose a state...")
        with col2:
            category = st.selectbox('Select Food Category', ["Processed Food", "Raw Food"],index=None,placeholder="Choose a category...")
            with col3:
                if category is None:
                    name = st.selectbox('Select Item',["Select Category First"],disabled=True)
                elif category == "Processed Food":
                    name = st.selectbox('Select Item', ["COCA COLA (BOTOL),1.5 liter","COCA COLA (TIN),320ml", "DUTCH LADY UHT COKLAT (KOTAK),200ml","DUTCH LADY UHT FULL CREAM (KOTAK),200ml"
                                                ,"F&N OREN (BOTOL),1.5 liter","F&N OREN (TIN),325 ml","HORLICKS (PAKET),400g","KICAP LEMAK MANIS CAP KIPAS UDANG,345ml"
                                                ,"KICAP MASIN ADABI,340ml","KICAP MANIS ADABI,340ml","KORDIAL SUNQUICK (OREN),840 ml","KRIMER MANIS PEKAT CAP SAJI,500g"
                                                ,"KRIMER SEJAT CAP F&N,390g","MACKAREL CAP AYAM (SOS TOMATO),425g","MAGGI MI SEGERA PERISA KARI,5 X 79g","MARJERIN DAISY,240g"
                                                ,"MENTEGA ANCHOR (SALTED),227g","MENTEGA KACANG HALUS LADY'S CHOIE,340g","MI SEDAP MI GORENG PERISA ASLI,5 X 90g"
                                                ,"MILO (PAKET),1kg","MILO (PAKET),400g","NESCAFE CLASSIC (PAKET),200 g","NESTLE CERELAC BERAS - TIN,500g","NESTLE COFFEE-MATE,450g"
                                                ,"PERENCAH NASI GORENG IKAN BILIS SERI AJI,26g","SANTAN KELAPA JENAMA KARA,200ml","SARDIN CAP ADABI (SOS TOMATO DENGAN CILI),425g"
                                                ,"SARDIN CAP AYAM (SOS TOMATO),155g","SARDIN CAP AYAM (SOS TOMATO),425g","SARDIN CAP KING CUP (SOS TOMATO),155g"
                                                ,"SARDIN CAP KING CUP (SOS TOMATO),425g","SERBUK KARI AYAM DAN DAGING ADABI,250g","SERBUK KARI DAGING BABAS,250g"
                                                ,"SERBUK KARI IKAN ADABI,250g","SERBUK KURMA AYAM & DAGING ADABI,250g","SERBUK KURMA BABAS,125g","SERBUK NASI GORENG CINA ADABI,17g"
                                                ,"SERBUK PERENCAH SUP ADABI,250g","SOS CILI KIMBALL,340 g","SOS CILI MAGGI,340 g","SOS CILI MAGGI,500 g","SOS TIRAM MAGGI,340g"
                                                ,"SOS TOMATO MAGGI,325 g","SUSU TEPUNG SEGERA EVERYDAY,900g","TEH BOH (UNCANG),100 beg","TEH LIPTON (UNCANG),100 beg","TUNA CAP AYAM (TUNA MAYONNAISE),160 g"
                                                ,"YOGURT MARIGOLD (LOW FAT) (STRAWBERRY),130g"],index=None,placeholder="Choose an item...")
                    
                elif category == "Raw Food":
                    name = st.selectbox('Select Item', ["AYAM BERSIH - STANDARD",'BAWANG BESAR IMPORT (INDIA)','BAWANG BESAR KUNING OR HOLLAND','BAWANG KECIL MERAH BIASA IMPORT (INDIA)',
                             'BAWANG KECIL MERAH ROSE IMPORT (INDIA)','BAWANG PERAI (LEEK) IMPORT','BAWANG PUTIH IMPORT (CHINA)','BAYAM HIJAU', 'BAYAM MERAH','BETIK BIASA','BROKOLI','CILI AKAR HIJAU','CILI AKAR MERAH',
                             'CILI HIJAU','CILI KERING KERINTING (BERTANGKAI OR TIDAK BERTANGKAI)','DADA AYAM (CHICKEN KEEL) (1KG)','DAGING LEMBU TEMPATAN BAHAGIAN DAGING PAHA (KECUALI BATANG PINANG - TENDERLOIN)',
                             'DRAGON FRUIT MERAH','EPAL HIJAU GRANNY SMITH (SAIZ M)','HALIA BASAH (TUA)','IKAN BAWAL HITAM (ANTARA 2 HINGGA 5 EKOR SEKILOGRAM)','IKAN BILIS GRED B (KOPEK)','IKAN CENCARU (ANTARA 4 HINGGA 6 EKOR SEKILOGRAM)',
                             'IKAN GELAMA (ANTARA 5 HINGGA 10 EKOR SEKILOGRAM)','IKAN JENAHAK','IKAN JENAHAK (KEPINGAN)','IKAN KELI (ANTARA 2 HINGGA 5 EKOR SEKILOGRAM)','IKAN KEMBUNG (ANTARA 8 HINGGA 12 EKOR SEKILOGRAM)',
                             'IKAN KEMBUNG KECIL OR PELALING','IKAN KERAPU (ANTARA 2 HINGGA 5 EKOR SEKILOGRAM)','IKAN KERISI (ANTARA 5 HINGGA 10 EKOR SEKILOGRAM)','IKAN MABUNG (ANTARA 6 HINGGA 10 EKOR SEKILOGRAM)',
                             'IKAN PARI (KEPINGAN)','IKAN SELAR PELATA','IKAN SELAYANG OR SARDIN','IKAN SIAKAP (ANTARA 2 HINGGA 4 EKOR SEKILOGRAM)','IKAN TENGGIRI BATANG (ANTARA 1 HINGGA 2 EKOR SEKILOGRAM)','IKAN TENGGIRI BATANG (KEPINGAN)',
                             'IKAN TILAPIA MERAH (ANTARA 2 HINGGA 5 EKOR SEKILOGRAM)','IKAN TONGKOL OR AYA OR KAYU HITAM (ANTARA 1 HINGGA 2 EKOR SEKILOGRAM)','JAMBU BATU BERBIJI','JAMBU BATU TANPA BIJI','KACANG BENDI','KACANG BOTOL','KACANG BUNCIS',
                             'KACANG DAL (AUSTRALIA)','KACANG HIJAU (IMPORT)','KACANG MERAH (IMPORT)','KACANG PANJANG','KACANG SOYA (IMPORT)','KACANG TANAH (IMPORT)','KAILAN','KANGKUNG','KELAPA BIJI','KELAPA PARUT','KEPAK AYAM (CHICKEN WING)','KEPALA IKAN JENAHAK',
                             'KERANG (SAIZ SEDERHANA)','KETAM RENJONG atau BUNGA(ANTARA 5 HINGGA 8 EKOR SEKILOGRAM)','KUBIS BULAT (TEMPATAN)' ,'KUBIS BULAT IMPORT (BEIJING)','KUBIS BULAT IMPORT (CHINA)','KUBIS BUNGA (CAULIFLOWER)', 'KUBIS PANJANG CHINA - BESAR',
                             'KUNYIT HIDUP', 'LADA BENGGALA HIJAU (CAPSICUM)','LADA BENGGALA KUNING (CAPSICUM)', 'LADA BENGGALA MERAH (CAPSICUM)','LENGKUAS','LIMAU KASTURI', 'LIMAU NIPIS', 'LOBAK MERAH', 'NENAS BIASA' ,'PAHA AYAM (CHICKEN DRUMSTICK)', 'PISANG BERANGAN',
                             'PISANG EMAS', 'SADERI','SANTAN KELAPA SEGAR (PEKAT)','SAWI HIJAU','SOTONG (Lebih atau samadengan 6 EKOR SEKILOGRAM)','TAUGE KACANG HIJAU' ,'TAUHU (JENIS KERAS)','TELUR AYAM GRED A','TELUR AYAM GRED B','TELUR AYAM GRED C','TELUR AYAM KAMPUNG',
                             'TELUR MASIN (1 biji)','TELUR MASIN (4 biji)','TEMBIKAI MERAH TANPA BIJI','TEMBIKAI SUSU','TEMPE (BUNGKUSAN PLASTIK)','TERUNG PANJANG','THIGH AYAM','TIMUN','TOMATO','UBI KENTANG IMPORT (CHINA)',
                             'UDANG HARIMAU (ANTARA 20 HINGGA 30 EKOR SEKILOGRAM)','UDANG KERING','UDANG PUTIH OR VANNAMEI (TERNAK) (ANTARA 41 HINGGA 60 EKOR SEKILOGRAM)','WHOLE LEG AYAM'],index=None,placeholder="Choose an item...")
    
    
        dummy_fig = go.Figure()
        dummy_date = pd.to_datetime('2023-01-01')
        dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='markers', name='Actual Price'))
        dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='lines', name='Predicted Price', line=dict(color='#e74c3c')))
        dummy_fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            title='Select a state and an item to view the Trend',
            showlegend=True,
        )
    
        if ((chosen_state is None) or (name is None) or (category is None)):
    
            st.plotly_chart(dummy_fig)
        else:
            if category=="Processed Food":
                percentage_difference=DisplayGraphProcessed(chosen_state, name)
            elif category=="Raw Food":
                percentage_difference=DisplayGraphRaw(chosen_state, name)
    
            if ((chosen_state is not None) and (name is not None) and (category is not None) ):
                if percentage_difference is not None:
        
                    if percentage_difference > 0:
                        st.write(f"The price is predicted to increase :red[+{str(round(percentage_difference,2))}%] in December 2023")
                    elif percentage_difference < 0:
                        st.write(f"The price is predicted to decrease :blue[{str(round(percentage_difference,2))}%] in December 2023")

            # MovingAverage(chosen_state,name,category)
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

