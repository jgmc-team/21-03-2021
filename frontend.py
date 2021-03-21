import pandas as pd
import os
# import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# %%
pd.set_option.max_rows = 1000


path = "./data/"
tmp = pd.DataFrame()
df = pd.DataFrame()
for city in os.listdir(path):
    tmp = pd.DataFrame(pd.read_json(path + city))
    df = pd.concat([df, tmp], axis=1)

# df.shape

CITIES = ['Алмазный',
          'Западный',
          'Курортный',
          'Лесной',
          'Научный',
          'Полярный',
          'Портовый',
          'Приморский',
          'Садовый',
          'Северный',
          'Степной',
          'Таежный',
          'Южный']

plt.plot(df.iloc[:365, 0])

PERIOD_20_Y = df.iloc[:, 1]
PERIOD_1_Y = df.iloc[:365, 1]
# PERIOD_20_Y.shape, PERIOD_1_Y.shape, PERIOD_1_Y.min(), PERIOD_1_Y.max()

# CITY_20_Y = pd.DataFrame(np.array(PERIOD_20_Y).reshape(365, -1))
# CITY_20_Y.head(5)

CITY_20_Y = pd.DataFrame(np.array(PERIOD_20_Y).reshape(-1, 365)).T
# CITY_20_Y.head(5)


# CITY_20_Y.shape

# %%
df_desc = CITY_20_Y.T.describe()
df_desc.head()

# %%
df_desc.loc['mean'].min(), df_desc.loc['mean'].max(), df_desc.shape

# %%
CITY_20_Y['mean'] = df_desc.loc['mean'].T

# %%
CITY_20_Y.head()

# %%
# CITY_20_Y['max'] = np.max(np.array(CITY_20_Y), axis = 1)
# CITY_20_Y['mean'] = np.mean(np.array(CITY_20_Y), axis = 1)
# CITY_20_Y['median'] = np.mean(np.array(CITY_20_Y), axis = 1)
# CITY_20_Y['std'] = np.std(np.array(CITY_20_Y), axis = 1)

# %%
# CITY_20_Y.shape

# %%
plt.plot(CITY_20_Y.iloc[0, :-5])

# %%
df = CITY_20_Y.copy()

# %%
y = df.pop('mean')
X = df.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# %%
prediction = rf.predict(X)

# %%
# np.mean(y - prediction)

import dash_html_components as html
import dash_core_components as dcc
import dash

# %%
app = dash.Dash(__name__)

# %%
app.layout = html.Div(
    children=[
        dcc.Dropdown(options=[{'label': i, 'value': i} for i in CITIES]),
        html.H1(children="Погода по городам", ),
        html.P(
            children="Анализ прогноза погоды"
            ,
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df.index,
                        "y": y,
                        "type": "lines",
                    },
                ],
                "layout": {"title": "Город:"},
            },
        )
    ]
)


if __name__ == "__main__":
    app.run_server(debug=False)

