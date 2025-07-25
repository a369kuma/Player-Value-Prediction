# player_value_prediction.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify
import schedule
import time

# ======================
# DATA COLLECTION
# ======================
def scrape_transfermarkt(url):
    """Scrape player data from Transfermarkt"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Scraping data from {url}...")
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    players = []
    table = soup.find('table', {'class': 'items'})
    
    for row in table.find_all('tr')[1:]:  # Skip header
        cols = row.find_all('td')
        if len(cols) > 1:
            player = {
                'name': cols[1].get_text(strip=True),
                'age': int(cols[2].get_text(strip=True)),
                'position': cols[3].get_text(strip=True),
                'market_value': cols[4].get_text(strip=True),
                'club': cols[5].get_text(strip=True),
                'league': cols[6].get_text(strip=True),
                # Additional stats you might want to add:
                'goals': int(cols[7].get_text(strip=True)) if len(cols) > 7 else 0,
                'assists': int(cols[8].get_text(strip=True)) if len(cols) > 8 else 0,
                'games_played': int(cols[9].get_text(strip=True)) if len(cols) > 9 else 0
            }
            players.append(player)
    
    return pd.DataFrame(players)

# ======================
# DATA PROCESSING
# ======================
def preprocess_data(df):
    """Clean and prepare the data for modeling"""
    
    # Clean market value
    df['market_value'] = (
        df['market_value']
        .str.replace('€', '')
        .str.replace('m', '000000')
        .str.replace('k', '000')
        .str.replace('[^0-9.]', '', regex=True)
    )
    df['market_value'] = pd.to_numeric(df['market_value'])
    
    # Encode categorical variables
    le = LabelEncoder()
    df['position_encoded'] = le.fit_transform(df['position'])
    df['league_encoded'] = le.fit_transform(df['league'])
    
    # Create additional features
    df['goals_per_game'] = df['goals'] / df['games_played'].replace(0, 1)
    df['assists_per_game'] = df['assists'] / df['games_played'].replace(0, 1)
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['age', 'goals', 'assists', 'games_played', 'goals_per_game', 'assists_per_game']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df, scaler, le

# ======================
# MODEL TRAINING
# ======================
def train_model(df):
    """Train and evaluate multiple models"""
    
    # Select features and target
    feature_cols = ['age', 'position_encoded', 'league_encoded', 
                   'goals', 'assists', 'games_played',
                   'goals_per_game', 'assists_per_game']
    X = df[feature_cols]
    y = df['market_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Train and evaluate
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {'MAE': mae, 'R2': r2}
        trained_models[name] = model
        
        print(f"{name} Results:")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R²: {r2:.4f}\n")
    
    return trained_models, results, feature_cols

# ======================
# MODEL DEPLOYMENT (API)
# ======================
def create_api(model, feature_cols, scaler=None, encoder=None):
    """Create a Flask API for predictions"""
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            
            # Prepare input features
            input_features = [
                data.get('age', 25),
                data.get('position_encoded', 0),
                data.get('league_encoded', 0),
                data.get('goals', 0),
                data.get('assists', 0),
                data.get('games_played', 0),
                data.get('goals_per_game', 0),
                data.get('assists_per_game', 0)
            ]
            
            # Scale features if scaler provided
            if scaler:
                input_features = scaler.transform([input_features])
            
            # Make prediction
            prediction = model.predict([input_features])[0]
            
            return jsonify({
                'status': 'success',
                'predicted_value': float(prediction),
                'currency': 'EUR'
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    return app

# ======================
# DATA UPDATE PIPELINE
# ======================
def update_pipeline(model_path='player_value_model.pkl', url=None):
    """Periodically update the model with fresh data"""
    if not url:
        url = 'https://www.transfermarkt.com/premier-league/marktwerte/wettbewerb/GB1'
    
    def update_task():
        print("Running scheduled update...")
        try:
            # 1. Get new data
            new_data = scrape_transfermarkt(url)
            
            # 2. Preprocess
            processed_data, _, _ = preprocess_data(new_data)
            
            # 3. Load current model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 4. Retrain with new data
            feature_cols = [col for col in processed_data.columns 
                          if col in ['age', 'position_encoded', 'league_encoded',
                                   'goals', 'assists', 'games_played',
                                   'goals_per_game', 'assists_per_game']]
            X = processed_data[feature_cols]
            y = processed_data['market_value']
            
            model.fit(X, y)
            
            # 5. Save updated model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print("Update completed successfully!")
        except Exception as e:
            print(f"Update failed: {str(e)}")
    
    # Schedule daily updates (could be weekly in production)
    schedule.every().day.at("02:00").do(update_task)
    
    print("Update scheduler started. Updates will run daily at 2 AM.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Example usage flow:
    
    # 1. Scrape and preprocess data
    print("=== DATA COLLECTION ===")
    url = "https://www.transfermarkt.com/premier-league/marktwerte/wettbewerb/GB1"
    df = scrape_transfermarkt(url)
    df, scaler, encoder = preprocess_data(df)
    
    # 2. Train models
    print("\n=== MODEL TRAINING ===")
    models, results, feature_cols = train_model(df)
    
    # 3. Save the best model (XGBoost in this case)
    best_model = models['XGBoost']
    with open('player_value_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # 4. Start API (in a real scenario, you'd run this separately)
    print("\n=== STARTING API ===")
    app = create_api(best_model, feature_cols, scaler, encoder)
    
    # In production, you would run the app separately:
    # app.run(host='0.0.0.0', port=5000)
    
    # For demo purposes, we'll just print instructions:
    print("\nAPI READY TO START!")
    print("To run the API in production, uncomment the app.run() line")
    print("Then you can send POST requests to /predict with player data")
    
    # 5. Start update scheduler (would run continuously)
    # print("\n=== STARTING UPDATE SCHEDULER ===")
    # update_pipeline()