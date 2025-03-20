import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class FeatureGenerator(ABC):
    """Abstract base class for feature generators"""

    @abstractmethod
    def calculate_features(self, data):
        """Calculate features from data"""
        pass


class ProductFeatureGenerator(FeatureGenerator):
    """Generate product-level features"""

    def calculate_features(self, data):
        """
        Calculate product-level features:
        - sales_product: Sales of the product and the store
        - MA7_P: Moving average of sales in the past 7 days
        - LAG7_P: Sales from 7 days earlier
        """
        # Ensure date is in datetime format
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        # Group by product_id, store_id, and date to get sales_product
        product_features = data.groupby(['product_id', 'store_id', 'date'])['quantity'].sum().reset_index()
        product_features.rename(columns={'quantity': 'sales_product'}, inplace=True)

        # Sort by product_id, store_id, and date for time-based features
        product_features.sort_values(['product_id', 'store_id', 'date'], inplace=True)

        # Create a separate DataFrame with all dates for each product-store combination
        all_dates = pd.date_range(data['date'].min() - timedelta(days=7), data['date'].max())
        product_store_combinations = product_features[['product_id', 'store_id']].drop_duplicates()

        # Create cartesian product of product-store combinations and dates
        index_combinations = []
        for _, row in product_store_combinations.iterrows():
            for date in all_dates:
                index_combinations.append((row['product_id'], row['store_id'], date))

        complete_index = pd.MultiIndex.from_tuples(
            index_combinations,
            names=['product_id', 'store_id', 'date']
        )

        # Reindex and fill missing values with 0
        product_sales_ts = product_features.set_index(['product_id', 'store_id', 'date'])['sales_product']
        product_sales_complete = product_sales_ts.reindex(complete_index, fill_value=0).reset_index()

        # Calculate MA7_P - Moving average of past 7 days
        product_sales_complete['MA7_P'] = product_sales_complete.groupby(
            ['product_id', 'store_id']
        )['sales_product'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )

        # Calculate LAG7_P - Sales from 7 days earlier
        product_sales_complete['LAG7_P'] = product_sales_complete.groupby(
            ['product_id', 'store_id']
        )['sales_product'].transform(
            lambda x: x.shift(7)
        )

        # Replace NaN with 0
        product_sales_complete.fillna(0, inplace=True)

        # Convert date back to string format for consistency
        product_sales_complete['date'] = product_sales_complete['date'].dt.strftime('%Y-%m-%d')

        return product_sales_complete


class SalesAnalyzer:
    """Main class to analyze sales data"""

    def __init__(self, min_date=None, max_date=None, top=5):
        """Initialize with date range and top N for output"""
        self.min_date = min_date
        self.max_date = max_date
        self.top = top

        # Initialize feature generators
        self.product_feature_generator = ProductFeatureGenerator()

    # [load_data method from Step 1]
    def load_data(self):
        """Load and prepare data from CSV files"""
        # Load CSV files
        self.brand_df = pd.read_csv('brand.csv')
        self.product_df = pd.read_csv('product.csv')
        self.store_df = pd.read_csv('store.csv')
        self.sales_df = pd.read_csv('sales.csv')

        # Convert date to datetime format
        self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])

        # Map product to brand_id
        product_brand_map = self.product_df.set_index('id')['brand'].to_dict()
        self.brand_dict = {}
        for product_id, brand_name in product_brand_map.items():
            brand_id = self.brand_df[self.brand_df['name'] == brand_name]['id'].values[0]
            self.brand_dict[product_id] = brand_id

        # Rename columns to match expected output
        self.sales_df.rename(columns={'product': 'product_id', 'store': 'store_id'}, inplace=True)

        # Add brand_id to sales data
        self.sales_df['brand_id'] = self.sales_df['product_id'].map(self.brand_dict)

        return self.sales_df

    def filter_date_range(self, data):
        """Filter data based on date range"""
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        if self.min_date:
            data = data[data['date'] >= self.min_date]

        if self.max_date:
            data = data[data['date'] <= self.max_date]

        return data

    def generate_product_features(self):
        """Generate product-level features"""
        # Load and prepare data
        data = self.load_data()

        # Filter by date range if specified
        if self.min_date or self.max_date:
            data = self.filter_date_range(data)

        # Generate product features
        product_features = self.product_feature_generator.calculate_features(data)

        return product_features


if __name__ == '__main__':
    # Create analyzer with default dates
    min_date = datetime.strptime('2021-01-08', '%Y-%m-%d')
    max_date = datetime.strptime('2021-05-30', '%Y-%m-%d')
    analyzer = SalesAnalyzer(min_date=min_date, max_date=max_date)

    # Generate product features
    product_features = analyzer.generate_product_features()
    print("Product features generated successfully")
    print(product_features.head())
