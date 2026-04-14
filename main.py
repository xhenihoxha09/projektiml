import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler

# Load
df = pd.read_csv('ted_main.csv')

# 1. Feature Engineering: Flatten Ratings
df['ratings_list'] = df['ratings'].apply(ast.literal_eval)
# Example: Extracting 'Funny' count
df['funny_count'] = df['ratings_list'].apply(lambda x: next((i['count'] for i in x if i['name'] == 'Funny'), 0))

# 2. Log Transform skewed columns
df['views_log'] = np.log1p(df['views'])

# 3. Scaling
scaler = StandardScaler()
features = ['views_log', 'comments', 'duration', 'languages', 'funny_count']
df_scaled = scaler.fit_transform(df[features])