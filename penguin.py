import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import skfuzzy as skfuzzy
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns


data = pd.read_csv('penguin.csv')

st.title("Clustering Fitur Berdasarkan Species")

st.header("Isi Dataset")
st.write(data)

data.rename(index=str, columns={
    'bill_length_mm' : 'bill length',
    'bill_depth_mm' : 'bill depth',
    'flipper_length_mm' : 'flipper length',
    'body_mass_g' : 'body mass (gr)'
}, inplace=True)

st.header("Menghapus Column Yang Tidak Digunakan")
data_fixed = data.drop(['island','sex','diet','life_stage','health_metrics','year'], axis=1)
data_fixed

X = data_fixed.iloc[:, 1:5].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

r = skfuzzy.cmeans(data = X.T, c = 3, m = 2, error = 0.0001, maxiter = 1000, init=None)

previsoes_porcentagem = r[1]
# previsoes_porcentagem
# previsoes_porcentagem[0][0]
# previsoes_porcentagem[1][0]
# previsoes_porcentagem[2][0]
# previsoes_porcentagem[0][0] + previsoes_porcentagem[1][0] +previsoes_porcentagem[2][0]
previsoes = previsoes_porcentagem.argmax(axis=0)
# previsoes

# Assuming 'previsoes' is defined somewhere in your code
column_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# Create a figure to hold the subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Mapping column indices to their corresponding labels
column_labels = {1: 'Bill Length', 2: 'Bill Depth', 3: 'Flipper Length', 4: 'Body Mass'}

from IPython.display import display
data_fixed['Cluster_Labels'] = previsoes
datafixed = data_fixed.sort_values(by='bill length', ascending=True)
st.write(data_fixed)

st.sidebar.subheader("Jumlah Cluster")
clust = st.sidebar.slider("Pilih Jumlah Cluster : ", 2,10,3,1)

st.header("Visualisasi Clustering Species Penguin")

def f_cmeans(n_clust):

    for k, (col1, col2) in enumerate(column_pairs):
        # Select the columns for the current combination
        X = data_fixed.iloc[:, [col1, col2]].values

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Perform fuzzy clustering
        r = skfuzzy.cmeans(data=X.T, c=n_clust, m=2, error=0.0001, maxiter=10000, init=None)

        # Assign clusters
        previsoes = np.argmax(r[1], axis=0)

        # Create a DataFrame for the data
        df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'species': previsoes})

        # Scatter plot for the current combination using seaborn
        sns.scatterplot(data=df, x='X1', y='X2', hue='species', palette='viridis', ax=axs[k // 3, k % 3], s=100)

        axs[k // 3, k % 3].set_title(f'Scatter Plot for Columns {col1} and {col2}')
        axs[k // 3, k % 3].set_xlabel(column_labels[col1])
        axs[k // 3, k % 3].set_ylabel(column_labels[col2])

    # Show the plots
    st.pyplot(plt)

f_cmeans(clust)


sns.pairplot(data_fixed[['bill length','bill depth','flipper length','body mass (gr)','species','Cluster_Labels']], hue='species')
st.pyplot(plt)












