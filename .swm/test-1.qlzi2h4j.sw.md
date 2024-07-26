---
title: test 1
---
<SwmSnippet path="/Dashboard.py" line="14">

---

this is for gettingsdsadasdasd the current price of an item and display it

```python
def CurrentPrice(name,category):
    if category == "Processed Food":
        data = pd.read_csv("data/processed food/"+name+".csv",index_col=False)
    elif category == "Raw Food":
        data = pd.read_csv("data/raw food/"+name+".csv",index_col=False)
    data['Percentage Difference (from OCT 2023)'] = data.groupby('state')['price'].pct_change() * 100
    # Filtering only the latest date prices
    df = data.groupby('state').tail(1)
    
    pos=df.nlargest(1,["Percentage Difference (from OCT 2023)"])
    neg=df.nsmallest(1,["Percentage Difference (from OCT 2023)"])
    
    df = df.rename(columns={'price': 'Price (NOV 2023)'})
    df = df.rename(columns={'state': 'State'})
    
    df['Price (NOV 2023)'] = 'RM ' + df['Price (NOV 2023)'].round(2).astype(str)

        
    df['Percentage Difference (from OCT 2023)'] = \
        df['Percentage Difference (from OCT 2023)'].apply(lambda x: f"{'+' if x > 0 else ''}{round(x, 2)}%{'📈' if x > 0 else ''}{'📉' if x < 0 else ''}" if pd.notna(x) else "")
                                    # .apply(lambda x: f"{'+' if x > 0 else ''}{round(x, 2)}%" if pd.notna(x) else "")
    
    # st.write("Percentage difference of price between October 2023 and November 2023 ")
    st.dataframe(df[['State',  'Price (NOV 2023)', 'Percentage Difference (from OCT 2023)']],hide_index=True, use_container_width=True,height=600)

    state_pos=pos['state'].iloc[0]
    state_neg=neg['state'].iloc[0]
    vary_pos=pos['Percentage Difference (from OCT 2023)'].iloc[0]
    vary_neg=neg['Percentage Difference (from OCT 2023)'].iloc[0]

    st.write(f"Most increment : :red[+{round(vary_pos,2)}%] in **{state_pos}** ")
    st.write(f"Most decrement : :blue[{round(vary_neg,2)}%] in **{state_neg}** ")

```

---

</SwmSnippet>

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBRllQJTNBJTNBR296ZW4yNA==" repo-name="FYP"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
