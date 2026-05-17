import pandas as pd

raw_data = {
    'Math': [85, 42, 91, 67, 55, 78],
    'Science': [70, 88, 60, 95, 40, 83],
    'English': [90, 76, 50, 88, 72, 61]
}
student_index = ['Ali', 'Buse', 'Cem', 'Deniz', 'Elif', 'Fatih']
scores = pd.DataFrame(raw_data, index=student_index)

print(scores)

loc_selection = scores.loc['Buse':'Elif', ['Math', 'Science']]
print(loc_selection)

iloc_selection = scores.iloc[-3:, :2]
print(iloc_selection)

boolean_mask = (scores['Math'] > 70) & (scores['English'] <= 80)
print(scores[boolean_mask])

scores.loc[scores['Math'] < 50, 'Science'] = 0
print(scores)

scores.drop(index='Cem', inplace=True)
scores.drop(columns='English', inplace=True)
print(scores)

extended_index = ['Ali', 'Buse', 'Deniz', 'Elif', 'Fatih', 'Gorkem']
scores = scores.reindex(extended_index, fill_value=0)
print(scores)

print(scores.isnull().sum())