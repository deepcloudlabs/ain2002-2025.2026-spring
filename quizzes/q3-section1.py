import pandas as pd

population_dict = {'Ankara': 5700000, 'Istanbul': 15900000, 'Izmir': 4400000, 'Bursa': 3100000}
population = pd.Series(population_dict)

print(population)
print("\nSeries Values Representation:", population.values)
print("Series Index Mapping:", population.index)
print("Series Data Type:", population.dtype)

target_labels = ['Adana', 'Ankara', 'Bursa', 'Istanbul']
pop_subset = population.reindex(target_labels)

print(pop_subset)
print(pd.isnull(pop_subset))

population_growth = population * 1.05
print(population_growth)

# Constructing the structural DataFrame representing multi-attribute data
city_data = {
    'city': ['Ankara', 'Istanbul', 'Izmir', 'Bursa'],
    'population': [5700000, 15900000, 4400000, 3100000],
    'area_km2': [25706, 5461, 12012, 10882]
}
city_df = pd.DataFrame(city_data)
print(city_df)

population_series_dict = city_df['population']
population_series_attr = city_df.population

print(population_series_dict)
print(population_series_attr)

istanbul_row_mask = city_df.loc[city_df['city'] == 'Istanbul']
print(istanbul_row_mask)

city_df['density'] = city_df['population'] / city_df['area_km2']
print(city_df)