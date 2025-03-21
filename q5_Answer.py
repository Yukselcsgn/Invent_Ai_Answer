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


class BrandFeatureGenerator(FeatureGenerator):
    """Generate brand-level features"""

    def calculate_features(self, data):
        """
        Calculate brand-level features:
        - sales_brand: Total sales of all products from the same brand and store
        - MA7_B: Moving average of brand sales in the past 7 days
        - LAG7_B: Brand sales from 7 days earlier
        """
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        brand_features = data.groupby(['brand_id', 'store_id', 'date'])['quantity'].sum().reset_index()
        brand_features.rename(columns={'quantity': 'sales_brand'}, inplace=True)

        brand_features.sort_values(['brand_id', 'store_id', 'date'], inplace=True)

        all_dates = pd.date_range(data['date'].min() - timedelta(days=7), data['date'].max())
        brand_store_combinations = brand_features[['brand_id', 'store_id']].drop_duplicates()

        index_combinations = []
        for _, row in brand_store_combinations.iterrows():
            for date in all_dates:
                index_combinations.append((row['brand_id'], row['store_id'], date))

        complete_index = pd.MultiIndex.from_tuples(
            index_combinations,
            names=['brand_id', 'store_id', 'date']
        )

        brand_sales_ts = brand_features.set_index(['brand_id', 'store_id', 'date'])['sales_brand']
        brand_sales_complete = brand_sales_ts.reindex(complete_index, fill_value=0).reset_index()

        brand_sales_complete['MA7_B'] = brand_sales_complete.groupby(
            ['brand_id', 'store_id']
        )['sales_brand'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )

        brand_sales_complete['LAG7_B'] = brand_sales_complete.groupby(
            ['brand_id', 'store_id']
        )['sales_brand'].transform(
            lambda x: x.shift(7)
        )

        brand_sales_complete.fillna(0, inplace=True)

        brand_sales_complete['date'] = brand_sales_complete['date'].dt.strftime('%Y-%m-%d')

        return brand_sales_complete


class StoreFeatureGenerator(FeatureGenerator):
    """Generate store-level features"""

    def calculate_features(self, data):
        """
        Calculate store-level features:
        - sales_store: Total sales of the store
        - MA7_S: Moving average of store sales in the past 7 days
        - LAG7_S: Store sales from 7 days earlier
        """

        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        store_features = data.groupby(['store_id', 'date'])['quantity'].sum().reset_index()
        store_features.rename(columns={'quantity': 'sales_store'}, inplace=True)

        store_features.sort_values(['store_id', 'date'], inplace=True)

        all_dates = pd.date_range(data['date'].min() - timedelta(days=7), data['date'].max())
        stores = store_features['store_id'].unique()

        index_combinations = []
        for store_id in stores:
            for date in all_dates:
                index_combinations.append((store_id, date))

        complete_index = pd.MultiIndex.from_tuples(
            index_combinations,
            names=['store_id', 'date']
        )

        store_sales_ts = store_features.set_index(['store_id', 'date'])['sales_store']
        store_sales_complete = store_sales_ts.reindex(complete_index, fill_value=0).reset_index()

        store_sales_complete['MA7_S'] = store_sales_complete.groupby(
            ['store_id']
        )['sales_store'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )

        store_sales_complete['LAG7_S'] = store_sales_complete.groupby(
            ['store_id']
        )['sales_store'].transform(
            lambda x: x.shift(7)
        )

        store_sales_complete.fillna(0, inplace=True)

        store_sales_complete['date'] = store_sales_complete['date'].dt.strftime('%Y-%m-%d')

        return store_sales_complete

class SalesAnalyzer:
    """Main class to analyze sales data"""

    def __init__(self, min_date=None, max_date=None, top=5):
        """Initialize with date range and top N for output"""
        self.min_date = min_date
        self.max_date = max_date
        self.top = top

        # Initialize feature generators
        self.product_feature_generator = ProductFeatureGenerator()
        self.brand_feature_generator = BrandFeatureGenerator()
        self.store_feature_generator = StoreFeatureGenerator()

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

    def generate_features(self):
        """Generate all features and combine them"""
        # Load and prepare data
        data = self.load_data()

        # Filter by date range if specified
        if self.min_date or self.max_date:
            data = self.filter_date_range(data)

        # Generate product features
        product_features = self.product_feature_generator.calculate_features(data)
        brand_features = self.brand_feature_generator.calculate_features(data)
        store_features = self.store_feature_generator.calculate_features(data)

        merged_df = pd.merge(
            product_features,
            brand_features,
            on=['store_id', 'date'],
            how='left'
        )
        merged_df = pd.merge(
            merged_df,
            store_features,
            on=['store_id', 'date'],
            how='left'
        )

        merged_df['brand_id'] = merged_df['product_id'].map(self.brand_dict)

        merged_df['date'] = pd.to_datetime(merged_df['date'])
        if self.min_date:
            merged_df = merged_df[merged_df['date'] >= self.min_date]
        if self.max_date:
            merged_df = merged_df[merged_df['date'] <= self.max_date]

        merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')

        merged_df.sort_values(['product_id', 'brand_id', 'store_id', 'date'], inplace=True)

        final_columns = [
            'product_id', 'store_id', 'brand_id', 'date', 'sales_product',
            'MA7_P', 'LAG7_P', 'sales_brand', 'MA7_B', 'LAG7_B',
            'sales_store', 'MA7_S', 'LAG7_S'
        ]

        self.features_df = merged_df[final_columns]

        return product_features

        def calculate_wmape(self):
            """Calculate WMAPE for each product-brand-store group"""
            # Group by product_id, store_id, brand_id
            groups = self.features_df.groupby(['product_id', 'store_id', 'brand_id'])

            # Calculate WMAPE for each group
            wmape_results = []
            for (product_id, store_id, brand_id), group in groups:
                # Skip groups with zero or NaN actuals
                if (group['sales_product'].sum() == 0 or pd.isna(group['sales_product']).all()):
                    continue

                actuals = group['sales_product'].values
                forecasts = group['MA7_P'].values

                # WMAPE calculation
                numerator = np.sum(np.abs(actuals - forecasts))
                denominator = np.sum(np.abs(actuals))

                if denominator > 0:
                    wmape = numerator / denominator
                    wmape_results.append({
                        'product_id': product_id,
                        'store_id': store_id,
                        'brand_id': brand_id,
                        'WMAPE': wmape
                    })

            # Create DataFrame and sort by WMAPE in descending order
            self.wmape_df = pd.DataFrame(wmape_results)
            self.wmape_df.sort_values('WMAPE', ascending=False, inplace=True)

            # Limit to top N results
            self.wmape_df = self.wmape_df.head(self.top)

            return self.wmape_df

        def run(self):
            """Run the complete analysis pipeline"""
            # Generate features
            self.generate_features()

            # Calculate WMAPE
            self.calculate_wmape()

            # Write results to CSV
            self.features_df.to_csv('features.csv', index=False)
            self.wmape_df.to_csv('mapes.csv', index=False)

            # Print preview of outputs
            print("\n--Output1 to be written to: features.csv--")
            print(f"[{','.join(self.features_df.columns)}]")
            print(self.features_df.head(2).to_string(index=False))
            print("...\n...")

            print("\n--Output2 to be written to: mapes.csv--")
            print(f"[{','.join(self.wmape_df.columns)}]")
            print(self.wmape_df.to_string(index=False))
            print("..\n..")

    def parse_arguments():
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='Sales data analysis')

        parser.add_argument(
            '--min-date',
            type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
            default=datetime.strptime('2021-01-08', '%Y-%m-%d'),
            help='Start date for analysis (YYYY-MM-DD)'
        )

        parser.add_argument(
            '--max-date',
            type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
            default=datetime.strptime('2021-05-30', '%Y-%m-%d'),
            help='End date for analysis (YYYY-MM-DD)'
        )

        parser.add_argument(
            '--top',
            type=int,
            default=5,
            help='Number of rows in WMAPE output'
        )

        return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    # Create analyzer and run pipeline
    analyzer = SalesAnalyzer(
        min_date=args.min_date,
        max_date=args.max_date,
        top=args.top
    )

    # Run analysis
    analyzer.run()