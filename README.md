# cancerDataset
Цель - спроектировать модель машинного обучения, которая будет по входным параметрам определять есть ли у человека рак груди.
## Предварительная обработка данных
### Загрузим данные из модуля datasets библиотеки Scikit-learn
```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
```
и посмотрим на стоставляющие датаесета
```python
cancer.keys()

>>> dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```
Создадим датафрейм, названия столбцов возьмем из cancer.feature_names и добавим целевую переменную target
```python
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target
```
Посчитаем у скольких людей значения target были 1 и у скольких 0
```python
unique, counts = np.unique(cancer.target, return_counts = True)

>>> (array([0, 1]), array([212, 357]))
```
### Нормализуем данные
Приведем все независимые переменные к единому масштабy, затем преобразуем обратно в датафрейм и вернем целевую переменную
```python
cancer_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
from sklearn.preprocessing import StandartScaler
scaler = StandartScaler()
scaler_data = scaler.fit_transform(cancer_df)

cancer_df_scaled = pd.DataFrame(scaled_data, columns = cancer.feature_names)
cancer_df_scaled['target'] = cancer.target
```
## Исследовательский анализ данных
Сгруппируем данные по целевой переменной, рассчитаем среднее и транспонируем датафрейм
```python
data = cancer_df_scaler.groupby('target').mean().T
```
Вычтем одну колонку из другой и посортируем по разнице
```python
data['diff'] = abs(data.iloc[:, 0]-data.iloc[:, 1])
data = data.sort_values(by = ['diff'], ascending = False)
```
Можно посмотреть на эти различия графически
