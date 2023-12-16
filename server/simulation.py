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

from data_loader import activity_month_distribution, length_probs, distribution_vectors_list, grouped_df, activity_group_value, save_df

# Calculate the probability distribution of activities across months
# activity_month_distribution = df_sale_12_HCM_22.groupby(['month', 'activity_group']).size().div(len(df_sale_12_HCM_22)).unstack().fillna(0)



# Calculate the probability distribution of gender dítribution
# gender_distribution = df_sale_12_HCM_22['gender'].value_counts(normalize=True)



# Distribution of cluster length
# lengths = []
# # Read each cluster file and calculate the gender distribution
# distribution_dataframes = []
# distribution_vectors_list = []

# df_cluster = pd.DataFrame()
# for i in range(0,10):
#   cluster_name = f"cluster_{i}"
#   df_cluster = pd.read_excel(f'./data/clusters/12m/cluster_{i}.xlsx')
#   lengths.append(len(df_cluster))
#   gender_distribution = df_cluster['gender'].value_counts(normalize=True).sort_index()
#   activity_distribution = df_cluster['activity_group'].value_counts(normalize=True).sort_index().fillna(0)
#   color_type_distribution = df_cluster['color_group'].value_counts(normalize=True).sort_index().fillna(0)
#   price_group_distribution = df_cluster['price_group'].value_counts(normalize=True).sort_index().fillna(0)

# # Append the distribution vectors to the list
# distribution_vectors_list.append([activity_distribution, color_type_distribution, price_group_distribution])

# Create a distribution dataframe
# distribution_df = pd.DataFrame(distribution_dataframes)

# #Create a probability distribution based on lengths
# length_probs = np.array(lengths) / sum(lengths)

# df_cluster = pd.read_excel('./data/clusters/12m/cluster_0.xlsx')
# the unique activity values list
# activity_group_value = df_cluster['activity_group'].drop_duplicates().reset_index(drop=True)



def get_product_count(attributes):
  count_for_combination = grouped_df.loc[
    (grouped_df['price_group'] == attributes[0]) &
    (grouped_df['color_group'] == attributes[1]) &
    (grouped_df['activity_group'] == attributes[2]) &
    (grouped_df['gender'] == attributes[3]),
    'count'
  ]
  if len(count_for_combination.values):
    return count_for_combination.values[0]
  return False


def update_product_count(attributes):
  product_count = get_product_count(attributes)
  if product_count:
    if product_count > 0:
      grouped_df.loc[
        (grouped_df['price_group'] == attributes[0]) &
        (grouped_df['color_group'] == attributes[1]) &
        (grouped_df['activity_group'] == attributes[2]) &
        (grouped_df['gender'] == attributes[3]),
        'count'
      ] = product_count - 1
      return True
    else:
      return False
  else:
    return False


def sample_data(cluster, group_code):
  group_distribution = distribution_vectors_list[cluster][group_code]
  sampled_group = np.random.choice(group_distribution.index, p=group_distribution)
  return sampled_group



AVTIVITY_GROUP_NUMBER = 5

POPULATION_SIZE = 500
TOTAL_TIME = 5
START_DAY = 1
START_MONTH = 12

CLUSTER_NUMBER = 10
ACTIVITY_GROUP_CODE = 0
COLOR_GROUP_CODE = 1
PRICE_GROUP_CODE = 2
GENDER_GROUP_CODE = 3


class Consumer:
    def __init__(self, type):
        self.type = type
        self.gender = sample_data(type, GENDER_GROUP_CODE)
        self.activity_group_paid = np.zeros(AVTIVITY_GROUP_NUMBER)

    def make_buy(self, month):
      environtment_activity = sample_activity(month)
      if environtment_activity not in distribution_vectors_list[0][0].index:
        return
      else:
        p = distribution_vectors_list[0][0][environtment_activity]
        bernoulli_dist = bernoulli(p)
        if(bernoulli_dist.rvs() == 1):
          color_group = sample_data(self.type, COLOR_GROUP_CODE)
          environtment_activity
          price_group = sample_data(self.type, PRICE_GROUP_CODE)
          if update_product_count([price_group, color_group, environtment_activity, self.gender]):
            # print(activity_group_value)
            # print(environtment_activity)
            index_number = activity_group_value.index[activity_group_value['activity_group'] == environtment_activity]
            # print(index_number)
            self.activity_group_paid[index_number] = self.activity_group_paid[index_number] + 1

# Convert the grouped DataFrame to a dictionary
# result_dict = grouped_df.to_dict(orient='records')

def sample_cluster():
  return np.random.choice(CLUSTER_NUMBER, p=length_probs)

# def sample_gender(cluster_index):
#   return np.random.choice(distribution_df.columns, p=distribution_df.iloc[cluster_index].fillna(0))

def sample_population(pop_size):
  consumers = []
  for i in range(0, pop_size):
      consumers.append(Consumer(type = sample_cluster()))
  return consumers

def sample_activity(sampled_month):
  # Retrieve the probability distribution vector for the sampled month
  sampled_month_probabilities = activity_month_distribution.loc[sampled_month].values

  # Scale the vector to ensure it totals to one
  scaled_sampled_month_probabilities = sampled_month_probabilities / sampled_month_probabilities.sum()

  # Use the scaled probabilities to sample an activity
  return np.random.choice(activity_month_distribution.columns, p=scaled_sampled_month_probabilities)

def simulation(pop_size = POPULATION_SIZE,
            total_time = TOTAL_TIME,
            start_day = START_DAY,
            start_month = START_MONTH):
  
  now = start_day
  consumers = sample_population(pop_size)

  # initialize the result df
  unique_combinations = grouped_df[['price_group', 'color_group', 'activity_group','gender']].drop_duplicates()
#   print(unique_combinations)
  result_df = pd.DataFrame(columns=['day'] + list(unique_combinations.itertuples(index=False, name=None)))
  for i in range(total_time):
    now += 1
    month = (now // 30 - 1 +  start_month) % 12 + 1
    month = 20220* 100 + month if month >= 10 else 202200*10 + month
    for consumer in consumers:
      consumer.make_buy(month)
    grouped_df['current_count'] = grouped_df['count']
    print(grouped_df.set_index(list(unique_combinations))['current_count'].to_dict())
    result_df = pd.concat([result_df, pd.DataFrame({'day': now, **grouped_df.set_index(list(unique_combinations))['current_count'].to_dict()}, index=[1])])

  # result_df.set_index('day').plot(marker='o')
  # plt.title('Product Count Over 100 Days')
  # plt.xlabel('Day')
  # plt.ylabel('Product Count')
  # plt.legend(title='Attribute Combinations')
  # plt.show()

  # calculate the std dev
  std_dev = result_df.set_index('day').groupby(axis=1, level=0).std()

  # Select the top 10 product types with the highest standard deviation
  top_10_types = std_dev.mean().nlargest(10).index

  # Plot the results for the top 10 product types
  fig, ax = plt.subplots(figsize=(20, 10))
  result_df.set_index('day')[top_10_types].plot(marker='o', ax=ax)
  plt.title('Top 10 Product Types with Highest Fluctuations')
  plt.xlabel('Day')
  plt.ylabel('Product Count')
  plt.legend(title='Attribute Combinations')
#   plt.show()
  plt.savefig("./output/result.png")
  save_df(result_df, "result.xlsx")
  top = result_df.set_index('day')[top_10_types]
  top.columns = ['_'.join(map(str, col)) for col in top.columns]
  return top.to_dict()


if __name__ == "__main__":
    simulation()


