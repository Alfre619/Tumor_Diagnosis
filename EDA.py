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

# Convert target to a single column with diagnosis labels
y = y['Diagnosis']


# Función para crear gráficas de violín o de cajas
def plot_features(features_range, plot_type='violin'):
    # Standardize the features for the given range
    data_std = (X.iloc[:, features_range] - X.iloc[:, features_range].mean()) / X.iloc[:, features_range].std()
    # Concatenate diagnosis with the standardized features
    data = pd.concat([y, data_std], axis=1)

    # Melt the dataframe to use with seaborn
    data_melted = pd.melt(data, id_vars='Diagnosis', var_name='features', value_name='value')

    # Create the plot
    plt.figure(figsize=(10, 10))
    if plot_type == 'violin':
        sns.violinplot(x='features', y='value', hue='Diagnosis', data=data_melted, split=True, inner='quart')
    elif plot_type == 'box':
        sns.boxplot(x='features', y='value', hue='Diagnosis', data=data_melted)

    plt.xticks(rotation=45)
    plt.title(
        f'{plot_type.capitalize()} plot of Standardized Features by Diagnosis for features {features_range.start + 1} to {features_range.stop}')
    plt.show()


# Llamar a la función para las características de 10 a 19 y de 20 a 29
ranges = [slice(0, 10), slice(10, 20), slice(20, 30)]
for r in ranges:
    plot_features(r, 'violin')
    plot_features(r, 'box')


# Función para crear joint plots
data_std = (X - X.mean()) / X.std()
# Concatenate diagnosis with the standardized features
data = pd.concat([y, data_std], axis=1)

# Melt the dataframe to use with seaborn
data_melted = pd.melt(data, id_vars='Diagnosis', var_name='features', value_name='value')

def create_joint_plot(data_melted, x_variable, y_variable, kind='reg', color='#ce1414'):
    # Verificar si las variables existen en el DataFrame
    if x_variable not in data_melted.columns or y_variable not in data_melted.columns:
        print(f"Error: verifica que los nombres de las variables '{x_variable}' y '{y_variable}' existan en el DataFrame.")
        print("Nombres de columnas actuales en el DataFrame:", data_melted.columns)
        return

    # Generate a joint plot for specified variables
    sns.jointplot(data=data_melted, x=x_variable, y=y_variable, kind=kind, color=color)
    plt.show()

# Imprimir nombres de columnas para verificar
print(data_melted.columns)


# Cambia estos nombres por los correctos si es necesario
create_joint_plot(data_melted=X, x_variable='concavity3', y_variable='concave_points3', kind='reg', color='#ce1414')

sns.set(style='whitegrid', palette='muted')
# Función para crear joint plots
data_std = (X - X.mean()) / X.std()
# Concatenate diagnosis with the standardized features
data = pd.concat([y, data_std.iloc[:, 0:10]], axis=1)
# Melt the dataframe to use with seaborn
data = pd.melt(data, id_vars='Diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10, 10))
sns.swarmplot(x='features', y='value', hue='Diagnosis', data=data)
plt.xticks(rotation=45);
plt.show()


f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax);
plt.show()