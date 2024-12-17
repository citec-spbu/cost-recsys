# TODO: Реализовать функцию выделение фичей датасета и реализует понедельное представление по каждому из товаров
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from src.data_selectoion import select_data

def process_data(orders: pd.DataFrame, purchases: pd.DataFrame, products: pd.DataFrame)-> pd.DataFrame:
    """
    Обрабатывает данные из трех DataFrame (orders, purchases, products)
    и возвращает обработанный DataFrame с понедельным представлением и выделенными признаками по каждому из товаров.

    Returns:
    - Результирующий DataFrame с понедельным представлением и выделенными признаками по каждому из товаров.
    """
    df = orders[['order_id', 'product_id', 'price']].copy()\
    .merge(purchases[['order_id', 'order_purchase_timestamp']].copy(), how='left', on='order_id')\
    .merge(products[['product_id', 'product_category_name']].copy(), how='left', on='product_id')
    df = select_data(df)
    df['week'] = df['order_purchase_timestamp'].dt.to_period('W').dt.start_time
    df.drop("order_purchase_timestamp", axis=1, inplace=True)
    df.drop("order_id", axis=1, inplace=True)
    df = (
    df
    .groupby(['week', 'product_id'])
    .agg(
        purchase_count=('price', 'size'),  # Count the number of rows (purchases)
        average_price=('price', 'mean')    # Calculate the average price
    )
    .reset_index()
    )
    # Perform one-hot encoding for the 'product_id' column
    one_hot_encoded = pd.get_dummies(df["product_id"], prefix="product")

    # Combine the one-hot encoded data with the original dataset
    df = pd.concat([df.reset_index(drop=True), one_hot_encoded], axis=1)
    df.drop("product_id", axis=1, inplace=True)
    rename_dict = {
    'product_06edb72f1e0c64b14c5b79353f7abea3': 'prod_A',
    'product_84f456958365164420cfc80fbe4c7fab': 'prod_B',
    'product_99a4788cb24856965c36a24e339b6058': 'prod_C',
    'product_ec2d43cc59763ec91694573b31f1c29a': 'prod_D',
    'product_f1c7f353075ce59d8a6f3cf58f419c9c': 'prod_E',
    }

    df.rename(columns=rename_dict, inplace=True)
    # Convert 'week' column to a datetime object if it's not already
    df['week'] = pd.to_datetime(df['week'])

    # Extract the day of the year (1-365) from the 'week' column
    df['day_of_year'] = df['week'].dt.dayofyear

    # Calculate angles for each day in a counterclockwise direction
    df['angle'] = 2 * np.pi * (df['day_of_year'] / 365)

    # Calculate x and y coordinates on the unit circle
    df['x_coordinate'] = np.sin(df['angle'])
    df['y_coordinate'] = np.cos(df['angle'])

    # Drop intermediate 'angle' column if not needed
    df.drop("angle", axis=1, inplace=True)
    df.drop("day_of_year", axis=1, inplace=True)
    df['week'] = pd.factorize(df['week'])[0]
    # Identify product columns
    product_cols = ['prod_A', 'prod_B', 'prod_C', 'prod_D', 'prod_E']

    # Step 1: Melt the dataframe so that product info is in rows instead of columns
    df_melted = pd.melt(
        df,
        id_vars=['week', 'purchase_count', 'average_price', 'x_coordinate', 'y_coordinate'],
        value_vars=product_cols,
        var_name='product',
        value_name='purchased' 
    )

    # Step 2: Filter to keep only rows where the product was actually purchased
    df_melted = df_melted[df_melted['purchased'] == True]

    # Step 3: Pivot the table to get product-specific purchase_count and average_price as columns
    df_pivoted = df_melted.pivot(
        index=['week', 'x_coordinate', 'y_coordinate'],
        columns='product',
        values=['purchase_count', 'average_price']
    )

    # Step 4: Flatten the MultiIndex columns
    df_pivoted.columns = [f'{metric}_{prod}' for metric, prod in df_pivoted.columns]

    # Step 5: Reset index to get a clean dataframe
    df = df_pivoted.reset_index()
    products = [col.split('_')[-1] for col in df.columns if col.startswith('purchase_count_prod_')]
    global_avg = df_melted[df_melted['purchased'] == True].groupby('product')['average_price'].mean()

    for product in products:
        purchase_col = f'purchase_count_prod_{product}'
        avg_price_col = f'average_price_prod_{product}'
        df[purchase_col] = df[purchase_col].fillna(0)
        df[avg_price_col] = df[avg_price_col].fillna(global_avg[f'prod_{product}'])
    return df