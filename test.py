import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example DataFrame (replace this with your actual data loading code)
data = {'attribute1': ['A', 'B', 'A', 'C'],
        'attribute2': ['X', 'Y', 'Z', 'X'],
        'attribute3': ['High', 'Low', 'Medium', 'High'],
        'attribute4': ['Male', 'Female', 'Male', 'Female'],
        'initial_count': [100, 150, 200, 120]}
product_df = pd.DataFrame(data)

# Get unique combinations of attributes
unique_combinations = product_df[['attribute1', 'attribute2', 'attribute3', 'attribute4']].drop_duplicates()
print(unique_combinations)

# Select the first 10 unique combinations for plotting
selected_combinations = unique_combinations.head(2)

# Simulation parameters
num_days = 10

# Initialize the result DataFrame
result_df = pd.DataFrame(columns=['day'] + list(selected_combinations.itertuples(index=False, name=None)))

# Loop through each day
for day in range(1, num_days + 1):
    # Simulate changes in product counts (replace this with your actual simulation logic)
    changes = np.random.randint(-10, 11, len(selected_combinations))
    product_df['current_count']= product_df['initial_count'].head(2) + changes

    print(product_df.set_index(list(selected_combinations))['current_count'].head(2))

    # Record the state in the result DataFrame
    result_df = pd.concat([result_df, pd.DataFrame({'day': [day], **product_df.set_index(list(selected_combinations))['current_count'].to_dict()})])

# Plot the results
result_df.set_index('day').plot(marker='o')
plt.title('Product Count Over 100 Days (Top 10 Types)')
plt.xlabel('Day')
plt.ylabel('Product Count')
plt.legend(title='Attribute Combinations')
plt.show()
