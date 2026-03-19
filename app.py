"""
app.py — WanderMatch Flask Application
========================================
Routes:
  GET  /          → Home page with user preference form
  POST /recommend → Runs ML prediction, renders destination results
  GET  /about     → Model performance comparison page
"""

from flask import Flask, render_template, request
from model import load_and_merge, preprocess, train_models, get_recommendations
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# STARTUP — Train all models once when the app launches
# ─────────────────────────────────────────────
print("=" * 55)
print("  WanderMatch — Loading datasets and training models...")
print("=" * 55)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: Load and merge the 4 CSV datasets
merged_df, destinations_df = load_and_merge(data_dir=DATA_DIR)
print(f"  Dataset loaded. Shape: {merged_df.shape}")

# Step 2: Preprocess the merged data
X, y, ENCODERS, FEATURE_COLS = preprocess(merged_df)
print(f"  Preprocessing complete. Features: {X.shape[1]}")

# Step 3: Train all 4 ML models and capture accuracy metrics
print("  Training classifiers...")
MODELS = train_models(X, y)

# Build a simplified accuracy dict for the templates
ACCURACY = {
    name: {
        "accuracy":  info["accuracy"],
        "precision": info["precision"],
    }
    for name, info in MODELS.items()
}

print("=" * 55)
print("  All models ready. Starting Flask server.")
print("=" * 55)


# ─────────────────────────────────────────────
# ROUTE: Home Page
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Render the home page with the travel preference form."""
    return render_template("index.html")


# ─────────────────────────────────────────────
# ROUTE: Recommendation Engine
# ─────────────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    1. Parse form input from the user
    2. Transform input to match ML training features
    3. Run Random Forest prediction
    4. Return top-5 destination recommendations with model metrics
    """
    # Parse form values
    interests    = request.form.getlist("interests")      # multi-select list
    budget       = request.form.get("budget", "Medium")
    climate      = request.form.get("climate", "Moderate")
    travel_type  = request.form.get("travel_type", "Couple")
    gender       = request.form.get("gender", "Male")

    # Map travel_type to adult count
    adults_map   = {"Solo": 1, "Couple": 2, "Family": 3, "Group": 5}
    num_adults   = adults_map.get(travel_type, 2)
    num_children = 1 if travel_type == "Family" else 0

    user_input = {
        "interests":    interests if interests else ["Nature"],
        "budget":       budget,
        "climate":      climate,
        "travel_type":  travel_type,
        "gender":       gender,
        "adults":       num_adults,
        "children":     num_children,
    }

    # Get recommendations from the ML pipeline
    recommendations, predicted_type = get_recommendations(
        user_input, MODELS, ENCODERS, destinations_df, top_n=5
    )

    return render_template(
        "results.html",
        recommendations  = recommendations,
        predicted_type   = predicted_type,
        accuracy         = ACCURACY,
        user_prefs       = {
            "interests":   interests,
            "budget":      budget,
            "climate":     climate,
            "travel_type": travel_type,
        }
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)