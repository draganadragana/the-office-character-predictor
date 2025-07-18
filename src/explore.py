import pandas as pd
import re

# Loading Dataset
df = pd.read_csv('D:/0 FAKS/2 letnik/machine learning and data mining/The Office - Character Predictor/data/the-office-lines.csv')
df['Line'] = df['Line'].str.lower().replace(r"[^\w\s']", '', regex=True)  # Removing special characters
df['Line'] = df['Line'].replace(r'\[.*?\]', '', regex=True)  # Removing stage directions

# Calculating character distribution
line_counts = df['Character'].value_counts()
total_lines = df['Line'].notna().sum()
selected_chars = line_counts[line_counts / total_lines >= 0.05].index
print(f"Characters with >= 5% lines: {selected_chars.tolist()}")

# Sorting selected characters
print("Characters with >= 5% lines:")
print(selected_chars.sort_values().tolist())

# Saving selected data
df_selected = df[df['Character'].isin(selected_chars)]
df_selected.to_csv('D:/0 FAKS/2 letnik/machine learning and data mining/The Office - Character Predictor/data/selected-lines.csv', index=False)