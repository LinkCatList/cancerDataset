# cancerDataset
Проектирование модели машинного обучения, которая будет по входным параметрам определять есть ли у человека рак груди.
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
![2171008f884d00ffadb74724201b0fa1](https://github.com/LinkCatList/cancerDataset/blob/main/picture.png)
## Обучение и оценка качества модели
### Разделение на обучающую и тестовую выборки
Размер тестовой выборки составит 30%, точка отсчета для вопроизводимости результата 42
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
### Обучение модели и прогноз
Используем логистическую регрессию из модуля linear_model
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```
Предсказание на тестовой выборке:
```python
y_pred = model.predict(X_test)

>>> array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1,
       0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
       0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0,
       1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1,
       1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
```
### Оценка качества модели
Построим матрицу ошибок, передадим ей тестовые и прогнозные значения. Поменяем порядок так, чтобы злокачественные опухоли было положительным классов, для удобства создадим датафрейм.
```python
from sklearn.metrics import confusion_matrix
model_matrix = confusion_matrix(y_test, y_pred, labels = [1,0])
model_matrix_df = pd.DataFrame(model_matrix)
```
Посчитаем, сколько было предсказанных значений и сколько было фактически
```python
(array([0, 1]), array([ 65, 106]))  # предикт

(array([0, 1]), array([ 63, 108]))  # фактически
```

Рассчитаем долю правильных ответов
$round((61 + 104)/(61 + 104 + 2 + 4), 2) = 0.96$
