import pandas as pd

def concatenate_excel_files(files, output_folder):
    df1 = pd.read_excel(files[0])
    # Read Excel files
    # for file in files[1:]:
    df2 = pd.read_excel(files[1])
    df1 = pd.concat([df1, df2], ignore_index=True)

    # Concatenate DataFrames

    # Write the result to a new Excel file
    all_sale_path = output_folder + '//all_sales.csv'
    df1.to_csv(all_sale_path, index=False)
    print(f"Concatenation completed. Result saved to {all_sale_path}")


    # grouped = df1.groupby('customer_id')

    # # Create separate DataFrames for each group
    # result_dataframes = {group_name: group for group_name, group in grouped}


    # for category, dataframe in result_dataframes.items():
    #     output_file_path = f'{output_folder}//df_sales_of_{category}.xlsx'
    #     dataframe.to_excel(output_file_path, index=False)
    #     print(f"DataFrame for Category {category} saved to {output_file_path}")

# Example usage
path = 'C://Users//pv//Downloads//sales_and_inventory_snapshot_data//InventoryAndSale_snapshot_data'

files = []

for i in [2022,2023]:
    for j in range(1,8):
        file = path + f"//Sales_snapshot_data//TT T0{j}-{i}_split_1.xlsx"
        files.append(file)
        print(file)

for i in range(8, 10):
    file = path + f"//Sales_snapshot_data//TT T0{i}-2022_split_1.xlsx"
    files.append(file)
    print(file)

for i in range(10, 13):
    file = path + f"//Sales_snapshot_data//TT T{i}-2022_split_1.xlsx"
    files.append(file)
    print(file)

output_file_folder = path + '//result'

concatenate_excel_files(files, output_file_folder)
