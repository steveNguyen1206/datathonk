import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Example DataFrame
data = {
    'Location': ['City1', 'City2', 'City1', 'City2', 'City3', 'City3', 'City4', 'City4'],
    'Color': ['Red', 'Blue', 'Red', 'Blue', 'Green', 'Yellow', 'Green', 'Yellow'],
    'Size': ['Small', 'Medium', 'Small', 'Medium', 'Large', 'ExtraLarge', 'Large', 'ExtraLarge'],
    'Heel': ['Low', 'Medium', 'Low', 'Medium', 'High', 'Low', 'High', 'Medium']
}

df = pd.DataFrame(data)

# Categorical features to one-hot encode
categorical_features = ['Location', 'Color', 'Size', 'Heel']

# Create a pipeline with one-hot encoding for categorical features
categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the pipeline to the categorical features
transformed_categorical = categorical_transformer.fit_transform(df)

# Display the result
print(transformed_categorical)

# print(pd.get_dummies(df[categorical_features], dtype=int))
