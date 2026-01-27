import pandas as pd
import re

class TravelEngine:
    def __init__(self):
        # 1. LOAD: Read all 4 raw datasets
        try:
            self.places = pd.read_csv('data/places.csv')
            self.city_info = pd.read_csv('data/city.csv')
            self.costs = pd.read_csv('data/travel_cost.csv')
            self.travel_details = pd.read_csv('data/travel_details_dataset.csv')
            print("✔ Raw CSVs loaded.")
        except Exception as e:
            print(f"❌ Error loading files: {e}")

        # 2. TRANSFORM: Run Preprocessing
        self.prepare_data()

    def clean_cost(self, cost_str):
        """Converts '500 - 1000' strings to integer 500 for budget filtering."""
        if pd.isna(cost_str): return 0
        nums = re.findall(r'\d+', str(cost_str))
        return int(nums[0]) if nums else 0

    def prepare_data(self):
        # Clean the cost column in travel_cost.csv
        self.costs['Min_Budget'] = self.costs['Accomdation_Cost'].apply(self.clean_cost)
        
        # MERGE: Create a Master Table for filtering
        # Join City info with Costs and Accommodation types
        self.master_df = pd.merge(self.city_info, self.costs, on='City', how='left')
        
        # Fill empty descriptions to avoid errors
        self.master_df = self.master_df.fillna("Not Available")
        print("✅ Preprocessing Complete: Master Table Generated.")

    def get_filtered_suggestions(self, budget=None, month=None, stay_type=None):
        """The heart of the filter system."""
        df = self.master_df.copy()

        # Apply filters only if user provided input
        if budget and budget.strip():
            df = df[df['Min_Budget'] <= int(budget)]
        
        if month and month.strip():
            df = df[df['Best Time'].str.contains(month, case=False, na=False)]
            
        if stay_type and stay_type.strip():
            df = df[df['Accommodation type'].str.contains(stay_type, case=False, na=False)]

        # Return list of matching cities
        return df.to_dict('records')

    def get_city_places(self, city_name):
        """Get all individual places for the selected city."""
        return self.places[self.places['City'] == city_name].to_dict('records')