"""
model.py — WanderMatch ML Pipeline
====================================
This module handles:
  1. Loading and merging the 4 CSV datasets
  2. Preprocessing (encoding + scaling)
  3. Training 4 ML classifiers (Random Forest, Decision Tree, KNN, Logistic Regression)
  4. Making destination recommendations based on user input

Now uses City.csv (100 real Indian destinations) as the recommendation pool,
with hand-assigned types and unique popularity scores to avoid identical results.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# ─────────────────────────────────────────────
# CITY TYPE & POPULARITY MAP
# Each destination gets a unique type and popularity score
# ─────────────────────────────────────────────
# Format: "City": (Type, Popularity, BudgetTier)
# BudgetTier: "Low" = budget-friendly, "Medium" = mid-range, "High" = premium
CITY_TYPE_MAP = {
    "Manali":                    ("Adventure", 9.1, "Medium"),
    "Leh Ladakh":                ("Adventure", 9.4, "High"),
    "Coorg":                     ("Nature",    8.7, "Medium"),
    "Andaman":                   ("Beach",     9.3, "High"),
    "Lakshadweep":               ("Beach",     8.9, "High"),
    "Goa":                       ("Beach",     9.5, "Medium"),
    "Udaipur":                   ("Historical",9.2, "High"),
    "Srinagar":                  ("Nature",    9.0, "Medium"),
    "Gangtok":                   ("Nature",    8.5, "Medium"),
    "Munnar":                    ("Nature",    8.8, "Medium"),
    "Varkala":                   ("Beach",     8.4, "Low"),
    "Mcleodganj":                ("Nature",    8.1, "Low"),
    "Rishikesh":                 ("Adventure", 8.9, "Low"),
    "Alleppey":                  ("Nature",    8.6, "Medium"),
    "Darjeeling":                ("Nature",    8.7, "Low"),
    "Nainital":                  ("Nature",    8.3, "Low"),
    "Shimla":                    ("Nature",    8.5, "Medium"),
    "Ooty":                      ("Nature",    8.2, "Low"),
    "Jaipur":                    ("Historical",9.1, "Medium"),
    "Lonavala":                  ("Nature",    7.8, "Low"),
    "Mussoorie":                 ("Nature",    8.0, "Low"),
    "Kodaikanal":                ("Nature",    8.1, "Low"),
    "Dalhousie":                 ("Nature",    7.9, "Low"),
    "Pachmarhi":                 ("Nature",    7.7, "Low"),
    "Varanasi":                  ("Historical",9.0, "Low"),
    "Mumbai":                    ("City",      9.3, "High"),
    "Agra":                      ("Historical",9.2, "Medium"),
    "Kolkata":                   ("City",      8.6, "Low"),
    "Jodhpur":                   ("Historical",8.8, "Medium"),
    "Bangalore":                 ("City",      8.7, "Medium"),
    "Amritsar":                  ("Historical",9.0, "Low"),
    "Delhi":                     ("City",      9.1, "Medium"),
    "Jaisalmer":                 ("Historical",8.9, "Medium"),
    "Mount Abu":                 ("Nature",    8.0, "Low"),
    "Wayanad":                   ("Nature",    8.4, "Medium"),
    "Hyderabad":                 ("City",      8.8, "Medium"),
    "Pondicherry":               ("Beach",     8.5, "Low"),
    "Khajuraho":                 ("Historical",8.3, "Low"),
    "Chennai":                   ("City",      8.4, "Medium"),
    "Vaishno Devi":              ("Adventure", 8.7, "Low"),
    "Ajanta and Ellora Caves":   ("Historical",8.6, "Low"),
    "Haridwar":                  ("Historical",8.5, "Low"),
    "Kanyakumari":               ("Beach",     8.2, "Low"),
    "Pune":                      ("City",      8.3, "Medium"),
    "Kochi":                     ("City",      8.5, "Medium"),
    "Ahmedabad":                 ("City",      8.1, "Low"),
    "Kanha National Park":       ("Nature",    8.6, "High"),
    "Mysore":                    ("Historical",8.7, "Low"),
    "Chandigarh":                ("City",      7.9, "Low"),
    "Hampi":                     ("Historical",8.8, "Low"),
    "Gulmarg":                   ("Adventure", 9.0, "High"),
    "Almora":                    ("Nature",    7.8, "Low"),
    "Shirdi":                    ("Historical",8.0, "Low"),
    "Auli":                      ("Adventure", 8.8, "High"),
    "Madurai":                   ("Historical",8.4, "Low"),
    "Amarnath":                  ("Adventure", 8.5, "Medium"),
    "Bodh Gaya":                 ("Historical",8.3, "Low"),
    "Mahabaleshwar":             ("Nature",    8.2, "Medium"),
    "Visakhapatnam":             ("Beach",     8.3, "Medium"),
    "Kasol":                     ("Adventure", 8.6, "Low"),
    "Nashik":                    ("Historical",7.8, "Low"),
    "Tirupati":                  ("Historical",8.6, "Low"),
    "Ujjain":                    ("Historical",8.1, "Low"),
    "Jim Corbett National Park": ("Nature",    9.0, "High"),
    "Gwalior":                   ("Historical",8.0, "Low"),
    "Mathura":                   ("Historical",8.2, "Low"),
    "Jog Falls":                 ("Nature",    7.9, "Low"),
    "Alibaug":                   ("Beach",     7.8, "Medium"),
    "Rameshwaram":               ("Historical",8.3, "Low"),
    "Vrindavan":                 ("Historical",8.1, "Low"),
    "Coimbatore":                ("City",      7.7, "Low"),
    "Lucknow":                   ("City",      8.2, "Low"),
    "Digha":                     ("Beach",     7.6, "Low"),
    "Dharamshala":               ("Adventure", 8.4, "Low"),
    "Kovalam":                   ("Beach",     8.5, "Medium"),
    "Kaziranga National Park":   ("Nature",    8.9, "High"),
    "Madikeri":                  ("Nature",    8.0, "Low"),
    "Matheran":                  ("Nature",    7.8, "Low"),
    "Ranthambore":               ("Nature",    8.7, "High"),
    "Agartala":                  ("City",      7.5, "Low"),
    "Khandala":                  ("Nature",    7.7, "Low"),
    "Kalimpong":                 ("Nature",    7.9, "Low"),
    "Thanjavur":                 ("Historical",8.2, "Low"),
    "Bhubaneswar":               ("Historical",7.9, "Low"),
    "Ajmer":                     ("Historical",8.0, "Low"),
    "Aurangabad":                ("Historical",8.1, "Medium"),
    "Jammu":                     ("Adventure", 7.8, "Low"),
    "Dehradun":                  ("Nature",    7.9, "Low"),
    "Puri":                      ("Beach",     8.4, "Low"),
    "Cherrapunji":               ("Nature",    8.3, "Medium"),
    "Bikaner":                   ("Historical",7.9, "Low"),
    "Shimoga (Shivamogga)":      ("Nature",    7.8, "Low"),
    "Hogenakkal":                ("Nature",    7.7, "Low"),
    "Gir National Park":         ("Nature",    8.5, "High"),
    "Kasauli":                   ("Nature",    7.8, "Low"),
    "Pushkar":                   ("Historical",8.4, "Low"),
    "Chittorgarh":               ("Historical",8.3, "Low"),
    "Nahan":                     ("Nature",    7.6, "Low"),
    "Lavasa":                    ("City",      7.7, "Medium"),
    "Poovar":                    ("Beach",     8.1, "Medium"),
}

CITY_BUDGET_MAP = {
    "Leh Ladakh": "High", "Andaman": "High", "Lakshadweep": "High",
    "Gulmarg": "High", "Auli": "High", "Kanha National Park": "High",
    "Kaziranga National Park": "High", "Ranthambore": "High",
    "Jim Corbett National Park": "High", "Gir National Park": "High",
    "Mumbai": "High", "Udaipur": "High",
    "Manali": "Medium", "Goa": "Medium", "Srinagar": "Medium",
    "Rishikesh": "Medium", "Shimla": "Medium", "Jaipur": "Medium",
    "Agra": "Medium", "Delhi": "Medium", "Jodhpur": "Medium",
    "Jaisalmer": "Medium", "Mysore": "Medium", "Hampi": "Medium",
    "Wayanad": "Medium", "Alleppey": "Medium", "Munnar": "Medium",
    "Darjeeling": "Medium", "Gangtok": "Medium", "Coorg": "Medium",
    "Bangalore": "Medium", "Hyderabad": "Medium", "Kolkata": "Medium",
    "Amritsar": "Medium", "Varanasi": "Medium", "Pondicherry": "Medium",
    "Kovalam": "Medium", "Visakhapatnam": "Medium", "Kochi": "Medium",
    "Pune": "Medium", "Chennai": "Medium", "Alibaug": "Medium",
    "Mahabaleshwar": "Medium", "Amarnath": "Medium", "Poovar": "Medium",
    "Cherrapunji": "Medium", "Aurangabad": "Medium", "Lavasa": "Medium",
    "Varkala": "Low", "Mcleodganj": "Low", "Nainital": "Low",
    "Ooty": "Low", "Lonavala": "Low", "Mussoorie": "Low",
    "Kodaikanal": "Low", "Dalhousie": "Low", "Pachmarhi": "Low",
    "Khajuraho": "Low", "Kanyakumari": "Low", "Haridwar": "Low",
    "Ajanta and Ellora Caves": "Low", "Vaishno Devi": "Low",
    "Ahmedabad": "Low", "Chandigarh": "Low", "Almora": "Low",
    "Shirdi": "Low", "Madurai": "Low", "Bodh Gaya": "Low",
    "Kasol": "Low", "Nashik": "Low", "Tirupati": "Low",
    "Ujjain": "Low", "Gwalior": "Low", "Mathura": "Low",
    "Jog Falls": "Low", "Rameshwaram": "Low", "Vrindavan": "Low",
    "Coimbatore": "Low", "Lucknow": "Low", "Digha": "Low",
    "Dharamshala": "Low", "Madikeri": "Low", "Matheran": "Low",
    "Agartala": "Low", "Khandala": "Low", "Kalimpong": "Low",
    "Thanjavur": "Low", "Bhubaneswar": "Low", "Ajmer": "Low",
    "Jammu": "Low", "Dehradun": "Low", "Puri": "Low",
    "Bikaner": "Low", "Shimoga (Shivamogga)": "Low", "Hogenakkal": "Low",
    "Kasauli": "Low", "Pushkar": "Low", "Chittorgarh": "Low",
    "Nahan": "Low", "Mount Abu": "Low",
}

STATE_MAP = {
    "Manali": "Himachal Pradesh", "Leh Ladakh": "Ladakh", "Coorg": "Karnataka",
    "Andaman": "Andaman & Nicobar", "Lakshadweep": "Lakshadweep", "Goa": "Goa",
    "Udaipur": "Rajasthan", "Srinagar": "Jammu & Kashmir", "Gangtok": "Sikkim",
    "Munnar": "Kerala", "Varkala": "Kerala", "Mcleodganj": "Himachal Pradesh",
    "Rishikesh": "Uttarakhand", "Alleppey": "Kerala", "Darjeeling": "West Bengal",
    "Nainital": "Uttarakhand", "Shimla": "Himachal Pradesh", "Ooty": "Tamil Nadu",
    "Jaipur": "Rajasthan", "Lonavala": "Maharashtra", "Mussoorie": "Uttarakhand",
    "Kodaikanal": "Tamil Nadu", "Dalhousie": "Himachal Pradesh", "Pachmarhi": "Madhya Pradesh",
    "Varanasi": "Uttar Pradesh", "Mumbai": "Maharashtra", "Agra": "Uttar Pradesh",
    "Kolkata": "West Bengal", "Jodhpur": "Rajasthan", "Bangalore": "Karnataka",
    "Amritsar": "Punjab", "Delhi": "Delhi", "Jaisalmer": "Rajasthan",
    "Mount Abu": "Rajasthan", "Wayanad": "Kerala", "Hyderabad": "Telangana",
    "Pondicherry": "Puducherry", "Khajuraho": "Madhya Pradesh", "Chennai": "Tamil Nadu",
    "Vaishno Devi": "Jammu & Kashmir", "Ajanta and Ellora Caves": "Maharashtra",
    "Haridwar": "Uttarakhand", "Kanyakumari": "Tamil Nadu", "Pune": "Maharashtra",
    "Kochi": "Kerala", "Ahmedabad": "Gujarat", "Kanha National Park": "Madhya Pradesh",
    "Mysore": "Karnataka", "Chandigarh": "Chandigarh", "Hampi": "Karnataka",
    "Gulmarg": "Jammu & Kashmir", "Almora": "Uttarakhand", "Shirdi": "Maharashtra",
    "Auli": "Uttarakhand", "Madurai": "Tamil Nadu", "Amarnath": "Jammu & Kashmir",
    "Bodh Gaya": "Bihar", "Mahabaleshwar": "Maharashtra", "Visakhapatnam": "Andhra Pradesh",
    "Kasol": "Himachal Pradesh", "Nashik": "Maharashtra", "Tirupati": "Andhra Pradesh",
    "Ujjain": "Madhya Pradesh", "Jim Corbett National Park": "Uttarakhand",
    "Gwalior": "Madhya Pradesh", "Mathura": "Uttar Pradesh", "Jog Falls": "Karnataka",
    "Alibaug": "Maharashtra", "Rameshwaram": "Tamil Nadu", "Vrindavan": "Uttar Pradesh",
    "Coimbatore": "Tamil Nadu", "Lucknow": "Uttar Pradesh", "Digha": "West Bengal",
    "Dharamshala": "Himachal Pradesh", "Kovalam": "Kerala",
    "Kaziranga National Park": "Assam", "Madikeri": "Karnataka", "Matheran": "Maharashtra",
    "Ranthambore": "Rajasthan", "Agartala": "Tripura", "Khandala": "Maharashtra",
    "Kalimpong": "West Bengal", "Thanjavur": "Tamil Nadu", "Bhubaneswar": "Odisha",
    "Ajmer": "Rajasthan", "Aurangabad": "Maharashtra", "Jammu": "Jammu & Kashmir",
    "Dehradun": "Uttarakhand", "Puri": "Odisha", "Cherrapunji": "Meghalaya",
    "Bikaner": "Rajasthan", "Shimoga (Shivamogga)": "Karnataka",
    "Hogenakkal": "Tamil Nadu", "Gir National Park": "Gujarat",
    "Kasauli": "Himachal Pradesh", "Pushkar": "Rajasthan", "Chittorgarh": "Rajasthan",
    "Nahan": "Himachal Pradesh", "Lavasa": "Maharashtra", "Poovar": "Kerala",
}


# ─────────────────────────────────────────────
# STEP 1 — LOAD & MERGE DATA
# ─────────────────────────────────────────────
def load_and_merge(data_dir="."):
    dest_path    = os.path.join(data_dir, "Expanded_Destinations.csv")
    users_path   = os.path.join(data_dir, "Final_Updated_Expanded_Users.csv")
    reviews_path = os.path.join(data_dir, "Final_Updated_Expanded_Reviews.csv")
    history_path = os.path.join(data_dir, "Final_Updated_Expanded_UserHistory.csv")
    city_path    = os.path.join(data_dir, "City.csv")

    destinations = pd.read_csv(dest_path)
    users        = pd.read_csv(users_path)
    reviews      = pd.read_csv(reviews_path)
    history      = pd.read_csv(history_path)

    merged = history.merge(destinations, on="DestinationID", how="inner")
    merged = merged.merge(users, on="UserID", how="inner")

    avg_ratings = reviews.groupby("DestinationID")["Rating"].mean().reset_index()
    avg_ratings.rename(columns={"Rating": "AvgRating"}, inplace=True)
    merged = merged.merge(avg_ratings, on="DestinationID", how="left")
    merged["AvgRating"] = merged["AvgRating"].fillna(merged["AvgRating"].median())

    print(f"  Merged shape: {merged.shape}")

    # Build enriched destination pool from City.csv
    city_df = pd.read_csv(city_path)
    rich_dests = []
    for _, row in city_df.iterrows():
        city_name = row["City"]
        best_time = row["Best Time"]
        dest_type, popularity, budget_tier = CITY_TYPE_MAP.get(city_name, ("City", 7.5, "Medium"))
        state = STATE_MAP.get(city_name, "India")
        rich_dests.append({
            "Name":            city_name,
            "State":           state,
            "Type":            dest_type,
            "BestTimeToVisit": best_time,
            "Popularity":      popularity,
            "BudgetTier":      budget_tier,
        })

    unique_dests = pd.DataFrame(rich_dests)
    print(f"  Enriched destination pool: {len(unique_dests)} destinations")
    return merged, unique_dests


# ─────────────────────────────────────────────
# STEP 2 — PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy().dropna()
    encoders = {}

    df["Preferences"] = df["Preferences"].apply(
        lambda x: [p.strip().capitalize() for p in str(x).split(",")]
    )
    mlb = MultiLabelBinarizer()
    pref_encoded = mlb.fit_transform(df["Preferences"])
    pref_df = pd.DataFrame(
        pref_encoded,
        columns=[f"pref_{c}" for c in mlb.classes_],
        index=df.index
    )
    encoders["mlb"]        = mlb
    encoders["mlb_classes"] = list(mlb.classes_)

    for col in ["Gender", "State", "BestTimeToVisit"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[f"le_{col}"] = le

    le_type = LabelEncoder()
    y = le_type.fit_transform(df["Type"].astype(str))
    encoders["le_Type"] = le_type

    numeric_cols = ["Popularity", "AvgRating", "ExperienceRating",
                    "NumberOfAdults", "NumberOfChildren"]
    cat_cols     = ["Gender", "State", "BestTimeToVisit"]

    X_base = df[numeric_cols + cat_cols].reset_index(drop=True)
    X = pd.concat([X_base, pref_df.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    encoders["scaler"]       = scaler
    encoders["numeric_cols"] = numeric_cols
    encoders["cat_cols"]     = cat_cols
    encoders["all_cols"]     = list(X.columns)

    return X.values, y, encoders, list(X.columns)


# ─────────────────────────────────────────────
# STEP 3 — TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    classifiers = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    }
    models = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2
        )
        models[name] = {"model": clf, "accuracy": acc, "precision": prec}
        print(f"  [{name}]  Accuracy: {acc}%  |  Precision: {prec}%")
    return models


# ─────────────────────────────────────────────
# STEP 4 — TRANSFORM USER INPUT
# ─────────────────────────────────────────────
def transform_user_input(user_input, encoders):
    budget_pop_map = {"Low": 7.8, "Medium": 8.4, "High": 9.0}
    travel_adults  = {"Solo": 1, "Couple": 2, "Family": 3, "Group": 5}
    climate_season = {
        "Tropical": "Nov-Mar", "Cold": "Nov-Feb",
        "Dry": "Oct-Mar",     "Moderate": "Sep-Mar",
    }

    popularity        = budget_pop_map.get(user_input.get("budget", "Medium"), 8.4)
    num_adults        = travel_adults.get(user_input.get("travel_type", "Couple"), 2)
    num_children      = int(user_input.get("children", 0))
    gender_str        = user_input.get("gender", "Male")
    avg_rating        = 3.0
    experience_rating = 3.0

    le_gender  = encoders["le_Gender"]
    gender_enc = le_gender.transform([gender_str])[0] if gender_str in le_gender.classes_ else 0
    state_enc  = 0

    btv_str = climate_season.get(user_input.get("climate", "Moderate"), "Sep-Mar")
    le_btv  = encoders["le_BestTimeToVisit"]
    btv_enc = le_btv.transform([btv_str])[0] if btv_str in le_btv.classes_ else 0

    mlb = encoders["mlb"]
    label_map = {
        "Nature": "Nature", "Adventure": "Adventure", "Culture": "Historical",
        "Relaxation": "Beaches", "Food": "City", "Heritage": "Historical",
        "Beaches": "Beaches", "Historical": "Historical", "City": "City",
    }
    raw_interests = [i.strip().capitalize() for i in user_input.get("interests", ["Nature"])]
    mapped = list({label_map.get(i, i) for i in raw_interests})
    known_classes = set(encoders["mlb_classes"])
    mapped = [m for m in mapped if m in known_classes]
    if not mapped:
        mapped = [encoders["mlb_classes"][0]]
    pref_enc = mlb.transform([mapped])[0]

    numeric_vals = [popularity, avg_rating, experience_rating, num_adults, num_children]
    cat_vals     = [gender_enc, state_enc, btv_enc]
    feature_vec  = np.array(numeric_vals + cat_vals + list(pref_enc), dtype=float).reshape(1, -1)

    n = len(encoders["numeric_cols"])
    feature_vec[:, :n] = encoders["scaler"].transform(feature_vec[:, :n])
    return feature_vec


# ─────────────────────────────────────────────
# STEP 5 — GET RECOMMENDATIONS
# ─────────────────────────────────────────────
def get_recommendations(user_input, models, encoders, unique_dests_df, top_n=5):
    feature_vec = transform_user_input(user_input, encoders)

    rf_model  = models["Random Forest"]["model"]
    le_type   = encoders["le_Type"]
    pred_enc  = rf_model.predict(feature_vec)[0]
    pred_type = le_type.inverse_transform([pred_enc])[0]

    budget_label = user_input.get("budget", "Medium")

    # Step 1: Best match — correct Type AND correct BudgetTier
    filtered = unique_dests_df[
        (unique_dests_df["Type"] == pred_type) &
        (unique_dests_df["BudgetTier"] == budget_label)
    ].copy().sort_values("Popularity", ascending=False)

    # Step 2: Not enough? Fill from same Type, any budget
    if len(filtered) < top_n:
        same_type_other = unique_dests_df[
            (unique_dests_df["Type"] == pred_type) &
            (unique_dests_df["BudgetTier"] != budget_label)
        ].sort_values("Popularity", ascending=False)
        filtered = pd.concat([filtered, same_type_other])

    # Step 3: Still not enough? Pull from matching budget, any type
    if len(filtered) < top_n:
        any_type_budget = unique_dests_df[
            unique_dests_df["BudgetTier"] == budget_label
        ].sort_values("Popularity", ascending=False)
        filtered = pd.concat([filtered, any_type_budget])

    filtered = filtered.drop_duplicates(subset="Name").head(top_n)

    recommendations = []
    for _, row in filtered.iterrows():
        pop_display = round((float(row["Popularity"]) - 7.0) / (9.5 - 7.0) * 100)
        recommendations.append({
            "name":        row["Name"],
            "type":        row["Type"],
            "state":       row["State"],
            "best_season": row["BestTimeToVisit"],
            "popularity":  pop_display,
            "budget":      budget_label,
            "booking_url": (
                "https://www.booking.com/search.html?ss="
                + row["Name"].replace(" ", "+")
            ),
        })
    return recommendations, pred_type


if __name__ == "__main__":
    print("Loading and merging datasets...")
    merged_df, unique_dests = load_and_merge()
    print("Preprocessing...")
    X, y, encoders, feature_cols = preprocess(merged_df)
    print(f"  Classes: {encoders['le_Type'].classes_}")
    print("Training models...")
    trained_models = train_models(X, y)
    test_input = {"interests": ["Nature", "Adventure"], "budget": "Medium",
                  "climate": "Cold", "travel_type": "Couple", "gender": "Male", "children": 0}
    recs, pred_type = get_recommendations(test_input, trained_models, encoders, unique_dests)
    print(f"\nPredicted type: {pred_type}")
    for r in recs:
        print(f"  -> {r['name']} ({r['state']}) | Pop={r['popularity']}/100 | {r['best_season']}")