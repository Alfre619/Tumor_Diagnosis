import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from ucimlrepo import fetch_ucirepo
# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)
# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# metadata
print(breast_cancer_wisconsin_diagnostic.metadata)

# variable information
print(breast_cancer_wisconsin_diagnostic)
print(X.describe(include='all'))
print(y.describe(include='all'))

# Contar las ocurrencias de cada categoría
counts = y.value_counts()
print('Number of Benign Tumors (B):', counts.get('B', 0))  # Usar get para evitar errores si no hay 'B'
print('Number of Malignant Tumors (M):', counts.get('M', 0))  # Usar get para evitar errores si no hay 'M'

counts.index = map(str, counts.index)
colors = ['blue', 'red']  # Azul para Benigno (B), Rojo para Maligno (M)
# Convertir los índices de counts a cadenas de texto si no lo son
# Crear una gráfica de barras usando matplotlib
plt.bar(counts.index, counts.values, color=colors)

# Etiquetar los ejes y dar un título al gráfico
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.title('Distribution of Benign (B) and Malignant (M) Diagnoses')

# Mostrar el gráfico
plt.show()
print(counts.index)
print(type(counts.index[0]))

print(X.describe())

