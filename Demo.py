import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:", layout="wide")

st.title(" :bar_chart: SuperStore Sales EDA")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding="ISO-8859-1")
else:
    # st.warning(r"Please upload a file to proceed")
    # st.stop()
    os.chdir(r"C:\Users\ADMIN\Documents\SalesProject")
    df = pd.read_csv("traincsvFormated(Edited3)Reduced.csv", encoding = "ISO-8859-1")
# Ensure all columns have compatible types for Arrow serialization
for column in df.columns:
    if df[column].dtype == 'object':
        try:
            df[column] = df[column].astype('string')
        except:
            df[column] = df[column].astype(str)

c1, c2 = st.columns((2))
#df["Order Date"] = pd.to_datetime(df["Order Date"],format='%d/%m/%Y', errors ='coerce')
df["Order Date"] = pd.to_datetime(df["Order Date"], format='%d/%m/%Y', errors='coerce')


# Getting the min and max date
startDate = pd.to_datetime(df["Order Date"],format='%d/%m/%Y', errors ='coerce').min()
endDate = pd.to_datetime(df["Order Date"],format='%d/%m/%Y', errors ='coerce').max()


with c1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with c2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()
#
#st.write("Columns after filtering by date:", df.columns.tolist())

st.sidebar.header("Choose your filter: ")
# create for country
country = st.sidebar.multiselect("Pick Your Country", df["Country"].unique())
if not country:
    df1 = df.copy()
else:
    df1 = df[df["Country"].isin(country)]    
# Create for Region
region = st.sidebar.multiselect("Pick your Region", df1["Region"].unique())
if not region:
    df2 = df1.copy()
else:
    df2 = df1[df1["Region"].isin(region)]

# Create for State
state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

# Create for City
city = st.sidebar.multiselect("Pick the City", df3["City"].unique())
if not city:
    df4 = df3.copy()
else:
    df4 = df3[df3["City"].isin(city)]    

# Filter the data based on Region, State and City and country

if not country and not region and not state and not city:
    filtered_df = df
elif not country:
    filtered_df = df1
elif not country and not city:
    filtered_df = df[df["Region"].isin(region)]    
elif not region and not city:
    filtered_df = df[df["State"].isin(state)]
elif not state and not city:
    filtered_df = df[df["Region"].isin(region)]
elif country and city:
    filtered_df = df3[df["Country"].isin(country) & df3["City"].isin(city)]    
elif region and city:
    filtered_df = df3[df["Region"].isin(state) & df3["City"].isin(city)]
elif state and city:
    filtered_df = df3[df["State"].isin(region) & df3["City"].isin(city)]
elif region and state:
    filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
elif country:
    filtered_df = df3[df3["Country"].isin(country) & df3["Region"].isin(region)]
elif region:
    filtered_df = df3[df3["Region"].isin(region)]    
elif city:
    filtered_df = df3[df3["City"].isin(city)]  
else:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city) & df3["Country"].isin(country)]


category_df = filtered_df.groupby(by=["Category"], as_index=False)["Sales"].sum()
#
subcategory_df = filtered_df.groupby(by=["Sub-Category"], as_index=False)["Sales"].sum()
#
city_df = filtered_df.groupby(by=["City"], as_index=False)["Sales"].sum()
#
region_df = filtered_df.groupby(by=["Region"], as_index=False)["Sales"].sum()
#
state_df = filtered_df.groupby(by=["State"], as_index=False)["Sales"].sum()
#
customer_df = filtered_df.groupby(by=["Customer ID"], as_index=False)["Sales"].sum()
#
filtered_df["month_year"] = filtered_df["Order Date"].dt.year
year_df = filtered_df.groupby(filtered_df["month_year"], as_index=False)["Sales"].sum()

# filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
# filtered_df["month_year"] = filtered_df["Order Date"].dt.year
# filtered_df['Order Date'] = pd.to_datetime(filtered_df['Order Date'], dayfirst=True)
#filtered_df['order_year'] = filtered_df['Order Date'].dt.year
# year_df = filtered_df.groupby(by=['Order Date'].dt.year)['Sales'].sum()
#year_df = filtered_df.groupby(filtered_df["month_year"])["Sales"].sum()
# chatgpt
# year_df = df["Order Date"].dt.year.groupby(df["Order Date"].dt.year)["Sales"].sum()
# year_df = df.groupby(df["Order Date"].dt.year)["Sales"].sum()
col1, col2 = st.columns((2))
with col1:
    st.subheader("Category wise Sales")
    fig = px.bar(category_df, x="Category", y="Sales", text=['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=150)

with col2:
    st.subheader("Region wise Sales")
    fig = px.pie(filtered_df, values="Sales", names="Region", hole=0.5)
    fig.update_traces(text=filtered_df["Region"], textposition="outside")
    st.plotly_chart(fig, use_container_width=True, height= 150)
col3, col4 = st.columns((2))
with col3:
    st.subheader("Yearly sales")
    fig = px.bar(year_df, x="month_year", y="Sales", text=['${:,.2f}'.format(x) for x in year_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=150)

with col4:
    st.subheader("Sub-Category wise Sales")
    fig = px.bar(subcategory_df, x="Sub-Category", y="Sales", text=['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=150)
#
col5, col6 = st.columns((2))
with col5:
    st.subheader("State wise Sales")
    fig = px.bar(state_df, x="State", y="Sales", text=['${:,.2f}'.format(x) for x in state_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=150)

with col6:
    st.subheader("Region wise Sales")
    fig = px.bar(region_df, x="Region", y="Sales", text=['${:,.2f}'.format(x) for x in region_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=150)

cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category_ViewData"):
        category = filtered_df.groupby(by="Category", as_index=False)["Sales"].sum()
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Category.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')

with cl2:
    with st.expander("Region_ViewData"):
        region = filtered_df.groupby(by="Region", as_index=False)["Sales"].sum()
        st.write(region_df.style.background_gradient(cmap="Oranges"))
        csv = region_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Region.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')
#
cl3, cl4= st.columns((2))

with cl3:
    with st.expander("State_ViewData"):
        city = filtered_df.groupby(by="State", as_index=False)["Sales"].sum()
        st.write(state_df.style.background_gradient(cmap="Blues"))
        csv = state_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="State.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')   
with cl4:
     with st.expander("Year_ViewData"):
        #year = filtered_df.groupby(by ="month_year", as_index = False)["Sales"].sum()
        year = filtered_df.groupby(by ="month_year", as_index = False)["Sales"].sum()
        st.write(year_df)
        csv = year_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Year.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file') 
        
      
cl5, cl6= st.columns((2))      
with cl5:
    with st.expander("City_ViewData"):
        city = filtered_df.groupby(by="City", as_index=False)["Sales"].sum()
        st.write(city_df.style.background_gradient(cmap="Blues"))
        csv = city_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="City.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file') 
  
with cl6:
    with st.expander("Customer_ViewData"):
        city = filtered_df.groupby(by="Customer ID", as_index=False)["Sales"].sum()
        st.write(customer_df.style.background_gradient(cmap="Blues"))
        csv = customer_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Customer.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file') 
        
# Time series analysis
filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader('Time Series Analysis')
linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')

# Create a treemap based on Region, category, sub-Category
st.subheader("Hierarchical view of Sales using TreeMap")
fig3 = px.treemap(filtered_df, path=["Region", "Category", "Sub-Category"], values="Sales", hover_data=["Sales"],
                  color="Sub-Category")
fig3.update_layout(width=800, height=650)
st.plotly_chart(fig3, use_container_width=True)

chart1, chart2 = st.columns((2))
with chart1:
    
    st.subheader("Segment wise sales")
    fig = px.pie(filtered_df, values="Sales", names="Segment", template="plotly_dark")
    fig.update_layout(width=600, height=600)
    fig.update_traces(text=filtered_df["Segment"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True, height= 120)

with chart2:
    
    st.subheader("Category wise sales")
    fig = px.pie(filtered_df, values="Sales", names="Category", template="gridon")
    fig.update_layout(width=600, height=600)
    fig.update_traces(text=filtered_df["Category"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True, height= 120)
#
# Shipment method
segment_df = filtered_df.groupby(by=["Ship Mode"], as_index=False)["Sales"].sum()
chart3, chart4 = st.columns((2))
with chart3:
    st.subheader("Category wise sales")
    fig = px.pie(filtered_df, values="Sales", names="Ship Mode", template="gridon")
    fig.update_layout(width=600, height=600)
    fig.update_traces(text=filtered_df["Ship Mode"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True, height= 120)

# Sales per state 
state_df = filtered_df.groupby(by=["State"], as_index=False)["Sales"].sum()
with chart4:
    st.subheader("State wise sales")
    fig = px.pie(filtered_df, values="Sales", names="State", template="gridon")
    fig.update_layout(width=600, height=600)
    fig.update_traces(text=filtered_df["State"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True, height= 120)

 #   
segment_df = filtered_df.groupby(by=["Segment"], as_index=False)["Sales"].sum()
with col1:
    st.subheader("Segment wise Sales")
    fig = px.bar(segment_df, x="Segment", y="Sales", text=['${:,.2f}'.format(x) for x in segment_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=150)

# Customer per state
state = filtered_df['State'].value_counts().reset_index()
state = state.rename(columns={'index':'State', 'State':'Number_of_customers'})
print(state.head(20))

# Customers per city 
city = filtered_df['City'].value_counts().reset_index()
city= city.rename(columns={'index':'City', 'City':'Number_of_customers'})
print(city.head(15))



# Group the data by state and calculate the total purchases (sales) for each state
state_sales = filtered_df.groupby(['State'])['Sales'].sum().reset_index()
# Sort the states based on their total sales in descending order to identify top spenders
top_sales = state_sales.sort_values(by='Sales', ascending=False)
# Print the states 
print(top_sales.head(20).reset_index(drop=True))

#sales per city

# Group the data by state and calculate the total purchase (sales) for each city
city_sales = filtered_df.groupby(['City'])['Sales'].sum().reset_index()
# Sort the cities based on their sales in descending order to identify top cities
top_city_sales = city_sales.sort_values(by='Sales', ascending=False)
# Print the states 
print(top_city_sales.head(20).reset_index(drop=True))

#sales per city $ state

state_city_sales = filtered_df.groupby(['State','City'])['Sales'].sum().reset_index()
print(state_city_sales.head(20))

#circle chart
# state_sales = df.groupby(['State'])['Sales'].sum().sort_values(ascending=False).head(10)
# fig = go.Figure(data=[go.Pie(labels=state_sales.index, values=state_sales.values)])
# fig.update_traces(textposition='inside', textinfo='percent+label')
#




#
import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region", "State", "City", "Category", "Sales", "Profit", "Quantity"]]
    fig = ff.create_table(df_sample, colorscale="Cividis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month wise sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
    sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))

# Create a scatter plot
data1 = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity")
data1['layout'].update(title="Relationship between Sales and Profits using Scatter Plot.",
                       titlefont=dict(size=20), xaxis=dict(title="Sales", titlefont=dict(size=19)),
                       yaxis=dict(title="Profit", titlefont=dict(size=19)))
st.plotly_chart(data1, use_container_width=True)

with st.expander("View Data"):
    st.write(filtered_df.iloc[:500, 1:20:2].style.background_gradient(cmap="Oranges"))

# Download original DataSet
csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")

