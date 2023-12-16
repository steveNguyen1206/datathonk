import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import bernoulli


from data_loader import LoadAllSale, save_df

# Separate numerical and categorical features
# selected features: month, gender, address2, price_group, color_group, activity_group, channel_id
numeric_features = []
categorical_features = [ 'gender', 'price_group', 'color_group', 'activity_group']

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
        # ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

features = numeric_features + categorical_features


def preProcess():
    df_sale_12_HCM_22 = LoadAllSale()
    df_sale_12_HCM_22['color_group'] = df_sale_12_HCM_22['color_group'].str.strip()
    # Combine numerical and one-hot encoded categorical features
    X = preprocessor.fit_transform(df_sale_12_HCM_22[features])
    return df_sale_12_HCM_22, X

#  Function to calculate the sum of squared distances
def calculate_squared_distances(X, k_values):
    squared_distances = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        squared_distances.append(kmeans.inertia_)
    return squared_distances

def getElbow():
    df_sale_12_HCM_22, X = preProcess()
    k_values = range(1, 20)
    # Calculate the sum of squared distances for each k
    squared_distances = calculate_squared_distances(X, k_values)
    # Plot the elbow method
    # plt.plot(k_values, squared_distances, marker='o')
    # plt.title('Elbow Method for Optimal k')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Sum of Squared Distances')
    # plt.show()
    return squared_distances

def Clustering(n_clusters = 10):

    df_sale_12_HCM_22, X = preProcess()

    df_clustered = df_sale_12_HCM_22[features]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    df_clustered['Cluster'] = kmeans.fit_predict(X)

    for cluster in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        save_df(cluster_data, f"cluster_{cluster}.xlsx")

    return None