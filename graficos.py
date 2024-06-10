import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/weatherhistoryszeged_80.csv")

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S.%f %z', utc=True)

df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df.hist(figsize=(12, 11))
plt.savefig('histograma.png')

columns_to_plot = ['Temperature', 'Apparent_Temperature', 'Humidity', 'Wind_Speed',
                   'Wind_Bearing', 'Visibility', 'Pressure']

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

for i, column in enumerate(columns_to_plot, start=1):
    plt.subplot(3, 3, i)
    sns.boxplot(df[column])
    plt.title(column)

plt.tight_layout()
plt.savefig('plot.png')


numeric = df.select_dtypes(include='float64')
#print(numeric.columns)

corr_matrix = numeric.corr()
#print(corr_matrix["Temperature"].sort_values(ascending=False))

numeric = numeric.drop('Loud_Cover', axis=1)

plt.figure(figsize=(14,10))
sns.heatmap(numeric.corr(), annot=True)
plt.savefig('heatmap.png')

numeric_data = df.select_dtypes(include=['float64', 'int64'])

df['Precip_Type']=df.apply(lambda row:'rain' if pd.isnull(row['Precip_Type']) and row['Temperature']>0 else ('snow' if pd.isnull(row['Precip_Type']) and row['Temperature']<=0 else row['Precip_Type']),axis=1)
df.drop_duplicates(inplace=True)

Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1

# Definir el umbral de outliers
threshold = 1.5

# Identificar outliers
outliers = (numeric_data < (Q1 - threshold * IQR)) | (numeric_data > (Q3 + threshold * IQR))


# Eliminar filas con outliers
df = df[~outliers.any(axis=1)]

columns_to_plot = ['Temperature', 'Apparent_Temperature', 'Humidity', 'Wind_Speed',
                   'Wind_Bearing', 'Visibility', 'Pressure']

## Creating boxplots for each column
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")  # Set the style of the plot

for i, column in enumerate(columns_to_plot, start=1):
    plt.subplot(3, 3, i)  # Create subplots
    sns.boxplot(df[column])
    plt.title(column)  # Set title for each subplot

plt.tight_layout()
plt.savefig('plot2.png')


quantitative_columns = ['Temperature', 'Apparent_Temperature', 'Humidity',
                        'Wind_Speed', 'Wind_Bearing', 'Visibility', 'Pressure']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, column in enumerate(quantitative_columns):
    sns.histplot(df[column], kde=True, ax=axes[i], alpha=0.7)
    axes[i].set_title(column)
    axes[i].set_xlabel('')

for j in range(len(quantitative_columns), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('distibution.png')

columns_to_plot = ['Temperature', 'Apparent_Temperature', 'Humidity', 'Wind_Speed',
                   'Wind_Bearing', 'Visibility', 'Pressure']

fig, axes = plt.subplots(nrows=1, ncols=len(columns_to_plot) - 1,
                         figsize=(6 * (len(columns_to_plot) - 1), 5))
feature_to_plot = 'Temperature'

# Loop over each feature (except the selected one) and create scatter plots
for i, col in enumerate(columns_to_plot):
    if col != feature_to_plot:
        sns.scatterplot(x=col, y=feature_to_plot, data=df, ax=axes[i if i < columns_to_plot.index(feature_to_plot) else i - 1])
        axes[i if i < columns_to_plot.index(feature_to_plot) else i - 1].set_xlabel(col)
        axes[i if i < columns_to_plot.index(feature_to_plot) else i - 1].set_ylabel(feature_to_plot)

plt.tight_layout()
plt.savefig('dispersion_values.png')


plt.figure(figsize=(10, 6))
heatmap_data = df.pivot_table(index='Year',columns='Month',values='Temperature')
sns.heatmap(data=heatmap_data,cmap='coolwarm')
plt.title('Heatmap of Temperature over time',fontsize=14,fontweight='bold')
plt.xlabel('Month',fontsize=14,fontweight='bold')
plt.ylabel('Year',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_overtime.png')

corr_matrix = numeric.corr()
print(corr_matrix["Temperature"].sort_values(ascending=False))

plt.figure(figsize=(14, 10))
sns.heatmap(numeric.corr(), annot=True, annot_kws={'size': 8})
plt.savefig('heatmapv2.png')
