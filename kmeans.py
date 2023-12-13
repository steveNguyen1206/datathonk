import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Example DataFrame
df = pd.read_excel('.//InventoryAndSale_snapshot_data//MasterData//Productmaster.xlsx')


# index	color	color_group	listing_price	price_group	gender	product_group	detail_product_group	shoe_product	size_group	size	age_group	activity_group	image_copyright	lifestyle_group	launch_season	mold_code	heel_height	code_lock	option	cost_price	product_id	product_syle_color	product_syle	brand_name	vendor_name


# Separate numerical and categorical features
numeric_features = ['listing_price', 'size', 'cost_price']
categorical_features = ['color', 'gender', 'product_group', 'shoe_product', 'age_group', 'activity_group', 'lifestyle_group', 'heel_height', 'product_syle']

# Create transformers for preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and k-means clustering
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('kmeans', KMeans(n_clusters=3, random_state=42))
# ])

# Fit the model
# pipeline.fit(df[features])

# Add cluster labels to the DataFrame
# df['Cluster'] = pipeline.named_steps['kmeans'].labels_

# Display the resulting DataFrame
# print(df)

#  Function to calculate the sum of squared distances
def calculate_squared_distances(X, k_values):
    squared_distances = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        squared_distances.append(kmeans.inertia_)
    return squared_distances

# Select relevant features for clustering
features = numeric_features + categorical_features

# Combine numerical and one-hot encoded categorical features
X = preprocessor.fit_transform(df[features])

# Specify a range of k values
k_values = range(1, 10)

# Calculate the sum of squared distances for each k
squared_distances = calculate_squared_distances(X, k_values)

# Plot the elbow method
plt.plot(k_values, squared_distances, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.show()
