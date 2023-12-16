import pandas as pd
import os


# df_sale_12_HCM_22 = pd.read_excel('./data/df_sale_12_22_HCM.xlsx')

grouped_df = pd.read_excel('data/2000_size_products.xlsx')

activity_month_distribution = pd.read_pickle('./data/activity_month_distribution.pkl')
# print(activity_month_distribution)

# distribution_vectors_df = pd.read_excel('./data/distribution_vector_df.xlsx')

length_probs = pd.read_excel('./data/length_probs.xlsx')[0].to_numpy()

activity_group_value = pd.read_excel('./data/activity_group_value.xlsx')

# Extract the list from the DataFrame
distribution_vectors_list = pd.read_pickle('./data/distribution_vectors.pkl')

def LoadAllSale():
    df_sale_12_HCM_22 = pd.read_excel('./data/df_sale_12_22_HCM.xlsx')
    return df_sale_12_HCM_22

def save_df(df, file_name):
    folder_path = './data'
    file_path = os.path.join(folder_path, file_name)
    df.to_excel(file_path, index=False)