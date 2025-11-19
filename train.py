import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 

# 1. Cargar el dataset
df = pd.read_csv('data.csv')

# 2. Selección de Variables
features = ['maxtemp', 'humidity', 'mean wind speed']
data_model = df[features].dropna()

# 3. Transformación de datos
data_log = np.log(data_model + 1)

# Escalado (StandardScaler)
scaler = StandardScaler()
scaler.fit(data_log)
data_sc = scaler.transform(data_log)

# Crear DataFrame normalizado para facilitar el uso
df_norm = pd.DataFrame(data_sc, columns=features)

# 4. Modelado (K-Means)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data_model['Cluster'] = kmeans.fit_predict(df_norm)

# 5. Visualización 3D
def plots_model():    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_model['maxtemp'], 
                         data_model['humidity'], 
                         data_model['mean wind speed'], 
                         c=data_model['Cluster'], 
                         cmap='viridis', 
                         s=50)
    
    ax.set_xlabel('Max Temp')
    ax.set_ylabel('Humidity')
    ax.set_zlabel('Mean Wind Speed')
    plt.title('Clustering Meteorológico 3D')
    
    # Añadir barra de color
    plt.colorbar(scatter)
    plt.show()

# Ejecutar la función de graficado
plots_model()

# Mostrar un ejemplo de los datos resultantes
print(data_model.head())