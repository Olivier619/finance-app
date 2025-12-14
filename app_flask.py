"""
API Flask pour servir les données de marché
Lancer avec: python app_flask.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from market_data import fetch_market_data, fetch_secteur_data

app = Flask(__name__)
CORS(app)

@app.route("/api/market-data")
def market_data():
    source = request.args.get("source", "yfinance")
    symbol = request.args.get("symbol", "AAPL")
    period = request.args.get("period", "1y")
    
    try:
        df = fetch_market_data(source, symbol, period=period)
        return jsonify({
            "success": True,
            "data": df.to_dict(orient="records"),
            "count": len(df)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/secteur-data")
def secteur_data():
    source = request.args.get("source", "yfinance")
    secteur = request.args.get("secteur", "technologie")
    
    try:
        df = fetch_secteur_data(secteur, source)
        return jsonify({
            "success": True,
            "data": df.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
