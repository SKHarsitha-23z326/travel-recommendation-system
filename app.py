from flask import Flask, render_template, request
from recommendation_engine import TravelEngine

app = Flask(__name__)
engine = TravelEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Capture user inputs
    budget = request.form.get('budget')
    month = request.form.get('month')
    stay = request.form.get('stay')
    
    # Get cities matching criteria
    results = engine.get_filtered_suggestions(budget, month, stay)
    return render_template('results.html', cities=results)

@app.route('/city/<name>')
def city_view(name):
    # Show detailed places for chosen city
    places = engine.get_city_places(name)
    return render_template('city_details.html', city=name, places=places)

if __name__ == '__main__':
    app.run(debug=True)