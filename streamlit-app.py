import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

# import data
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')
X.drop('Unnamed: 0', axis=1, inplace=True)
y.drop('Unnamed: 0', axis=1, inplace=True)
X = X.drop(['long', 'lat', 'electricity'], axis=1)


# title
st.header("""ISFEST 2023 : Final Data Competition - BEBAS""")

# layout
col1, col2 = st.columns([1,1], gap="large")
with col1:
    st.header("Model & Feature Selection")
    option = st.selectbox(
        'Choose Model',
        ('Select Model','Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'Desicion Tree', 'Random Forest')
    )
    # selected_columns = ['district', 'city', 'facilities', 'certificate', 'property_condition', 'building_size_m2', 'land_size_m2', 'maid_bathrooms','electricity', 'bedrooms','floors','carports','garages']
    building_size_m2 = st.number_input('Building Size (m2)', format='%f',value=272.0)
    land_size_m2 = st.number_input('Land Size (m2)', format='%f', value=239.0)
    maid_bathrooms = float(st.number_input('Maid Bathrooms', format='%d',value=1))
    maid_bedrooms = float(st.number_input('Maid Bedrooms', format='%d', value=0))
    bedrooms = float(st.number_input('Bedrooms', format='%d', value=2))
    floors = float(st.number_input('Floors', format='%d', value=2))
    carports = float(st.number_input('Carports', format='%d', value=0))
    garages = float(st.number_input('Garages', format='%d', value=0))




with col2:
    st.header("Prediction")
    try:
        if option == 'Linear Regression':
            st.write('You selected:', option)
            model = LinearRegression()
        elif option == 'Ridge Regression':
            st.write('You selected:', option)
            model = Ridge(alpha=10)
        elif option == 'Lasso Regression':
            st.write('You selected:', option)
            model = Lasso(alpha=0.001)
        elif option == 'KNN':
            st.write('You selected:', option)
            model = KNeighborsRegressor(n_neighbors=3)
        elif option == 'Desicion Tree':
            st.write('You selected:', option)
            model = DecisionTreeRegressor(max_depth=8)
        elif option == 'Random Forest':
            st.write('You selected:', option)
            model = RandomForestRegressor(n_estimators=100,
                                    random_state=3,
                                    max_samples=0.5,
                                    max_features=0.75,
                                    max_depth=15)
        # Splitting data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        # Melatih model pada set pelatihan
        # print(y)
        # print('test')
        model.fit(X, y)

        X_test = pd.DataFrame({'building_size_m2': [building_size_m2],
                                'land_size_m2': [land_size_m2],
                                'maid_bathrooms': [maid_bathrooms],
                                'maid_bedrooms': [maid_bedrooms],
                                'bedrooms': [bedrooms],
                                'floors': [floors],
                                'carports': [carports],
                                'garages': [garages]})
        # print(X_test)
        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)
        try:
            st.write('Predicted Price:', y_pred[0][0])
        except:
            st.write('Predicted Price:', y_pred[0])
    except:
        st.write('Please select a model.')