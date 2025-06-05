import streamlit as st
import pandas as pd
import joblib

# Загрузка моделей и фичей  
model = joblib.load('../models/car_price_model.pkl')
with open('../models/feature_list.txt', 'r') as f:
    feature_list = f.read().split(',')

st.title('🚗 Предсказание стоимости автомобиля')

# Форма для ввода данных
with st.form("car_form"):
    year = st.slider("Год выпуска", 1990, 2023, 2018)
    km_driven = st.number_input("Пробег (км)", 0, 500000, 50000)
    fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG"])
    transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
    owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner"])
    
    submitted = st.form_submit_button("Предсказать цену")
    
    if submitted:
        # Преобразование ввода в формат модели
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'transmission': [transmission],
            'owner': [owner]
        })
        
        # Предсказание
        prediction = model.predict(input_data)[0]
        st.success(f"### Предсказанная цена: ₹{prediction:,.2f}")

        # Визуализация
        st.subheader("Влияние параметров на цену")
        fig, ax = plt.subplots()
        features = ['year', 'mileage', 'engine_size', 'horsepower']
        importances = model.feature_importances_
        indices = np.argsort(importances)[-4:]
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_list[i] for i in indices])
        st.pyplot(fig)