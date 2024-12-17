# TODO: Реализовать функцию выделение товаров, на которых будет обучаться модель
import pandas as pd

def select_data(dataset: pd.DataFrame)-> pd.DataFrame:
    # Count the occurrences of each category
    category_counts = dataset["product_category_name"].value_counts()

    # Find the category with the most purchases
    most_purchased_category = category_counts.idxmax()
    dataset = dataset[dataset["product_category_name"] == most_purchased_category]
    # Count the occurrences of each product_id
    product_counts = dataset["product_id"].value_counts()

    # Get the top 5 product_ids with the most purchases
    top_5_products = product_counts.head(5).index

    # Filter the dataset to include only these top 5 products
    top_5_dataset = dataset[dataset["product_id"].isin(top_5_products)]

    top_5_dataset = top_5_dataset.reset_index(drop=True)
    top_5_dataset.drop("product_category_name", axis=1, inplace=True)

    return top_5_dataset