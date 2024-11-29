# Análise de Aprovação de Alunos

Modelo de regressão linear simples para prever aprovação de alunos comparando dados de 2015 e 2016.

## Funcionalidades

### Análise Exploratória
```python
# Visualizações com Seaborn
sns.histplot(data=df, x='ano_2015', kde=True)
sns.histplot(data=df, x='ano_2016', kde=True)
sns.regplot(x="ano_2015", y="ano_2016", data=df)
```

### Normalização
- MinMaxScaler
- StandardScaler
```python
scaler = MinMaxScaler()
df_normal = pd.DataFrame(scaler.fit_transform(df))

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df))
```

### Modelagem
```python
x = df_normal[["ano_2015"]]
y = df_normal[["ano_2016"]]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

lr = LinearRegression()
lr.fit(x_train, y_train)
```

## Métricas
- R² (Coeficiente de Determinação)
- MAE (Erro Médio Absoluto)
- MSE (Erro Quadrático Médio)
- RMSE (Raiz do Erro Quadrático Médio)

## Requisitos
```bash
pip install pandas numpy seaborn scikit-learn statsmodels
```

## Estrutura
- Script: `regressão_linear_simples_sklearn.py`
- Dataset: `aprovacao_alunos.xlsx`

## Uso
```bash
python regressão_linear_simples_sklearn.py
```

## Visualizações
- Histogramas de distribuição
- Gráfico de regressão
- Correlação entre anos
