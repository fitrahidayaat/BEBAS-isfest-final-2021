import streamlit as st
st.set_page_config(layout="wide")
# title
st.write("""
# ISFEST 2023 : Final Data Competition - BEBAS
## Modelling
""")

option = st.selectbox(
    'Choose Model',
    ('Select Model','Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'Desicion Tree', 'Random Forest'))



if option == 'Linear Regression':
    st.write('You selected:', option)
    code = """
        model = LinearRegression()
        # Melatih model pada set pelatihan
        model.fit(X_train, y_train)

        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)

        print('R2 score',r2_score(y_test,y_pred))
        print('MAE',mean_absolute_error(y_test,y_pred))
        print('MSE',mean_squared_error(y_test,y_pred))
        """
    st.code(code, language='python')
    st.code("""
        R2 score 0.8217216631864406
        MAE 483232540.55679226
        MSE 6.028820598411803e+17
    """, language='txt')
elif option == 'Ridge Regression':
    st.write('You selected:', option)
    code = """
        model = Ridge(alpha=10)
        # Melatih model pada set pelatihan
        model.fit(X_train, y_train)

        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)

        print('R2 score',r2_score(y_test,y_pred))
        print('MAE',mean_absolute_error(y_test,y_pred))
        print('MSE',mean_squared_error(y_test,y_pred))
        """
    st.code(code, language='python')
    st.code("""
        R2 score 0.8217164589239807
        MAE 483227540.2201481
        MSE 6.028996590432365e+17
    """, language='txt')
elif option == 'Lasso Regression':
    st.write('You selected:', option)
    code = """
        model = Lasso(alpha=0.001)
        # Melatih model pada set pelatihan
        model.fit(X_train, y_train)

        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)

        print('R2 score',r2_score(y_test,y_pred))
        print('MAE',mean_absolute_error(y_test,y_pred))
        print('MSE',mean_squared_error(y_test,y_pred))
        """
    st.code(code, language='python')
    st.code("""
        R2 score 0.8217216631864191
        MAE 483232540.5566044
        MSE 6.028820598412529e+17
    """, language='txt')
elif option == 'KNN':
    st.write('You selected:', option)
    code = """
        model = KNeighborsRegressor(n_neighbors=3)
        # Melatih model pada set pelatihan
        model.fit(X_train, y_train)

        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)

        print('R2 score',r2_score(y_test,y_pred))
        print('MAE',mean_absolute_error(y_test,y_pred))
        print('MSE',mean_squared_error(y_test,y_pred))
        """
    st.code(code, language='python')
    st.code("""
        R2 score 0.9907222150198749
        MAE 28574747.474747475
        MSE 3.1374592222222224e+16
    """, language='txt')
elif option == 'Desicion Tree':
    st.write('You selected:', option)
    code = """
        model = DecisionTreeRegressor(max_depth=8)
        # Melatih model pada set pelatihan
        model.fit(X_train, y_train)

        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)

        print('R2 score',r2_score(y_test,y_pred))
        print('MAE',mean_absolute_error(y_test,y_pred))
        print('MSE',mean_squared_error(y_test,y_pred))
        """
    st.code(code, language='python')
    st.code("""
        R2 score 0.9456465974534194
        MAE 247651745.60513726
        MSE 1.838063551205794e+17
    """, language='txt')
elif option == 'Random Forest':
    st.write('You selected:', option)
    code = """
        model = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)
        # Melatih model pada set pelatihan
        model.fit(X_train, y_train)

        # Membuat prediksi pada set pengujian
        y_pred = model.predict(X_test)

        print('R2 score',r2_score(y_test,y_pred))
        print('MAE',mean_absolute_error(y_test,y_pred))
        print('MSE',mean_squared_error(y_test,y_pred))
        """
    st.code(code, language='python')
    st.code("""
        R2 score 0.9945634384302784
        MAE 44576375.898374386
        MSE 1.838476562093377e+16
    """, language='txt')
else:
    st.write('Please select a model.')