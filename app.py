import streamlit as st
import pandas as pd
import os
from plots_func import plot_hist, plot_log_hist, plot_boxplot, plot_corr_num, plot_barplot, plot_positive_barplot_in_group, plot_positive_barplot

st.set_page_config(layout="wide", page_title="EDA Анализ")
st.title('Анализ данных по отклику на маркетинговую рекламу')
st.subheader('Полученные данные')

data = pd.read_csv('cleaned_full_data.csv')
data = data.drop('Unnamed: 0', axis =1)
st.dataframe(data)
st.subheader('Общая информация о данных')
data_describe = pd.read_csv("Data_overview.csv")

st.table(data_describe)


st.sidebar.title('Настройки')
feature_type = st.sidebar.radio('Тип признака:', ['Числовой', 'Категориальный'])

all_features =list(data.columns)
num_features = ['AGE', 'PERSONAL_INCOME', 'WORK_TIME']
cat_features = [feature for feature in all_features if feature not in num_features]

if feature_type == 'Числовой':
    num_col = st.sidebar.radio('Выберите числовой признак:', num_features)

    col_min = int(data[num_col].min())
    col_max = int(data[num_col].max())

    selected_range = st.sidebar.slider(
        'Выберите диапазон значений',
        min_value = col_min,
        max_value = col_max,
        value = (col_min, col_max)
    )

    filtered_data = data[(data[num_col] >= selected_range[0]) & (data[num_col] <= selected_range[1])]

    st.subheader(f'Анализ признака: {num_col} в диапазоне {selected_range[0]} - {selected_range[1]}')
    plot_type = st.selectbox('Выберите тип графика:', ['Boxplot', 'Histplot', 'Log Histplot'])
    if st.button('Показать распредления'):
        if plot_type == 'Boxplot':
            plot_boxplot(num_col, filtered_data)
        if plot_type == 'Histplot':
            plot_hist(num_col, filtered_data)
        if plot_type == 'Log Histplot':
            plot_log_hist(num_col, filtered_data)


    st.subheader("Проверка нулевой гипотезы: распределение признака не зависит от таргета")
    stat_test = pd.read_csv('NUM_features_stat_test.csv')
    st.table(stat_test)
    plot_corr_num(num_features, data)
    st.subheader('Вывод по числовым признакам: ')
    st.text('1. Между значениями числовых признаков в различных категориях таргета наблюдаются статистически значимые различия.')
    st.text('2. Связь между числовыми признаками и таргетной переменной является слабой, '
            'но статистически значимой. У признака PERSONAL_INCOME выявлена слабая прямая корреляция с таргетом, '
            'а у признаков AGE и WORK_TIME — слабая обратная связь.')
    st.text('3. Между числовыми признаками наблюдается низкий уровень взаимной корреляции, '
            'за исключением пары AGE и WORK_TIME, для которой выявлена умеренная положительная связь. '
            'Однако наличие прямой зависимости не установлено.')

if feature_type == 'Категориальный':
    cat_col = st.sidebar.radio('Выберите категориальный признак:', cat_features)
    st.subheader(f'Анализ признака: {cat_col}')
    plot_type = st.selectbox('Выберите тип графика:', ['Barplot', 'Доля положительного отклика по группам', 'Вклад категории в положительный отклик'])
    if st.button('Показать распредления'):
        if plot_type == 'Barplot':
            plot_barplot(cat_col, data)
        if plot_type == 'Доля положительного отклика по группам':
            plot_positive_barplot_in_group(cat_col, data)
        if plot_type == 'Вклад категории в положительный отклик':
            plot_positive_barplot(cat_col, data)

    cat_stat_tests = pd.read_csv('CAT features stats test.csv')
    st.subheader("Проверка нулевой гипотезы: распределение признака не зависит от таргета")
    st.table(cat_stat_tests.drop('Unnamed: 0', axis=1))
    st.subheader('Корреляция между категориальными признаками с помощью метода Крамера: ')
    st.image('Cramers_heatmap.png', use_container_width=True)
    st.subheader('Вывод по категориальным признакам:')
    st.text('1. Все категориальные признаки, представленные в наших данных можно разделить на несколько категорий: '
            'бинарные признаки (пол, статус работы, статус пенсионера, наличие квартиры), '
            'орядковые (образование, кол-во машин, кол-во детей, кол-во иждивенцев, доход в семье) '
            'и категориальные (регион, профессия, сфера деятельности и тд.)')
    st.text('2. Не все категориальные признаки демонстрируют статистически значимую связь с таргетной переменной '
            '(откликом на рекламную кампанию). Для части признаков различия между категориями не объясняются таргетом, '
            'что может говорить об их низкой информативности. Такие признаки потенциально можно исключить на этапе построения модели, '
            'чтобы снизить шум и переобучение.')


st.subheader('Связь признаков с таргетом:')
st.image('all_features_corr_coef.png', use_container_width=True)
st.text('\tПри расчёте коэффициентов зависимости между признаками и таргетной переменной была выявлена в целом слабая связь.')
st.text('\tМежду самими признаками значимой корреляции не обнаружено, за исключением пары AGE и WORK_TIME, '
        'где наблюдается умеренная положительная связь.')
st.text('\tДальнейшая работа будет заключаться в кодировании признаков и применении модели классического машинного обучения для решения задачи. ')
st.text('\tЗадача будет решаться следующим образом - алгоритм на основе полученных данных, будет предсказывать вероятность отклика клиента на рекламу.')