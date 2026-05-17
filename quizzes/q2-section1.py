import pandas as pd

population_dict = {'Ankara': 5700000, 'Istanbul': 15900000, 'Izmir': 4400000, 'Bursa': 3100000}
population = pd.Series(population_dict)

print(population)
print("\nSeries Values:", population.values)
print("Series Index:", population.index)
print("Series Data Type:", population.dtype)

target_labels = ['Adana', 'Ankara', 'Bursa', 'Istanbul']
pop_subset = population.reindex(target_labels)

print(pop_subset)
print(pd.isnull(pop_subset))

population_growth = population * 1.05
print(population_growth)