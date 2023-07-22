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
# print(y.shape)


# title
st.header("""ISFEST 2023 : Final Data Competition - BEBAS""")

# layout
col1, col2 = st.columns([1,2], gap="large")
with col1:
    st.header("Model & Feature Selection")
    option = st.selectbox(
        'Choose Model',
        ('Select Model','Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'Desicion Tree', 'Random Forest')
    )
    # selected_columns = ['district', 'city', 'facilities', 'certificate', 'property_condition', 'building_size_m2', 'land_size_m2', 'maid_bathrooms','electricity', 'bedrooms','floors','carports','garages']
    features = st.multiselect(
        'Select Features',
        X.columns
    )

with col2:
    st.header("Prediction")
    try:
        if option == 'Linear Regression':
            st.write('You selected:', option)
            model = LinearRegression()
            used_model = """LinearRegression()"""
        elif option == 'Ridge Regression':
            st.write('You selected:', option)
            model = Ridge(alpha=10)
            used_model = """Ridge(alpha=10)"""
        elif option == 'Lasso Regression':
            st.write('You selected:', option)
            model = Lasso(alpha=0.001)
            used_model = """Lasso(alpha=0.001)"""
        elif option == 'KNN':
            st.write('You selected:', option)
            model = KNeighborsRegressor(n_neighbors=3)
            used_model = """KNeighborsRegressor(n_neighbors=3)"""
        elif option == 'Desicion Tree':
            st.write('You selected:', option)
            model = DecisionTreeRegressor(max_depth=8)
            used_model = """DecisionTreeRegressor(max_depth=8)"""
        elif option == 'Random Forest':
            st.write('You selected:', option)
            model = RandomForestRegressor(n_estimators=100,
                                    random_state=3,
                                    max_samples=0.5,
                                    max_features=0.75,
                                    max_depth=15)
            used_model = """RandomForestRegressor(n_estimators=100,
                                    random_state=3,
                                    max_samples=0.5,
                                    max_features=0.75,
                                    max_depth=15)"""
        else:
            st.write('Please select a model.')
        
        code = f"""
                model = {used_model}
                # Melatih model pada set pelatihan
                model.fit(X_train, y_train)

                # Membuat prediksi pada set pengujian
                y_pred = model.predict(X_test)

                print('R2 score',r2_score(y_test,y_pred))
                print('MAE',mean_absolute_error(y_test,y_pred))
                print('MSE',mean_squared_error(y_test,y_pred))
                """
        st.code(code, language='python')

        # Splitting data
        X = X[features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Melatih model pada set pelatihan
        model.fit(X_train, y_train)

        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)

        st.write('R2 score',r2_score(y_test,y_pred))
        st.write('MAE',mean_absolute_error(y_test,y_pred))
        st.write('MSE',mean_squared_error(y_test,y_pred))
    except:
        st.write('Please select some features.')