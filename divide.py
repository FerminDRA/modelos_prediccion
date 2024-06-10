import pandas as pd

#data = pd.read_csv('usparticulatematte.csv')
df = pd.read_csv('data/weatherhistoryszeged.csv')

df=df.rename(columns={'Formatted Date':'Date','Precip Type':'Precip_Type',
                      'Temperature (C)':'Temperature',
                      'Apparent Temperature (C)':'Apparent_Temperature',
                     'Wind Speed (km/h)':'Wind_Speed',
                     'Wind Bearing (degrees)':'Wind_Bearing',
                     'Visibility (km)':'Visibility',
                     'Loud Cover':'Loud_Cover',
                     'Pressure (millibars)':'Pressure',
                     'Daily Summary':'Daily_Summary'})

data = df.fillna(0)
# Guardar el 100% de los datos
#data.to_csv('data/weatherHistory_100.csv', index=False)

#seventy_percent_size = int(len(data) * 0.20)
#seventy_percent_data = data.iloc[:seventy_percent_size]
#seventy_percent_data.to_csv('data/weatherhistoryszeged_20.csv', index=False)

# Guardar el 50% de los datos
#half_size = int(len(data) / 2)
#half_data = data.iloc[:half_size]
#half_data.to_csv('data/weatherHistory_50.csv', index=False)

# Guardar el 20% de los datos
#twenty_percent_size = int(len(data) * 0.80)
#twenty_percent_data = data.iloc[:twenty_percent_size]
#twenty_percent_data.to_csv('data/weatherhistoryszeged_80.csv', index=False)

first_part_proportion = 0.80
second_part_proportion = 0.20

# Calcular los tamaños de cada parte
first_part_size = int(len(data) * first_part_proportion)
second_part_size = len(data) - first_part_size

# Obtener las particiones
first_part_data = data.iloc[:first_part_size]
second_part_data = data.iloc[first_part_size:]

# Guardar el 20% de los datos
first_part_data.to_csv('data/weatherhistoryszeged_80.csv', index=False)

# Guardar el 80% de los datos
second_part_data.to_csv('data/weatherhistoryszeged_20.csv', index=False)