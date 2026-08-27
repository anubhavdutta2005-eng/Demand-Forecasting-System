import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.main import app, startup_event

# Initialize TestClient
client = TestClient(app)

def test_backend_flow():
    # 1. Trigger startup events to load model and data
    print("Triggering startup events...")
    startup_event()
    
    # 2. Test Health Endpoint
    print("\n--- Testing GET /health ---")
    response = client.get("/health")
    print(f"Status Code: {response.status_code}")
    print(f"Body: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["datasets_loaded"] is True
    print("Health check passed!")

    # 3. Test Model Info Endpoint
    print("\n--- Testing GET /model-info ---")
    response = client.get("/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Body: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert "Store" in data["features"]
    assert data["total_stores"] == 1115
    print("Model info check passed!")

    # 4. Test Store Details Endpoint
    print("\n--- Testing GET /store/1 ---")
    response = client.get("/store/1")
    print(f"Status Code: {response.status_code}")
    print(f"Body: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert data["store_id"] == 1
    assert data["store_type"] == "c"
    assert data["assortment"] == "a"
    print("Store details check passed!")

    # 5. Test Predict Endpoint (Standard June 2015 run)
    print("\n--- Testing POST /predict (Standard June 2015) ---")
    req_body = {
        "store_id": 1,
        "start_date": "2015-06-01",
        "end_date": "2015-06-07"
    }
    response = client.post("/predict", json=req_body)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    data = response.json()
    print(f"Predictions returned: {len(data['predictions'])}")
    print(f"Metrics: {data['metrics']}")
    
    # Assert predictions match expected days
    assert len(data["predictions"]) == 7
    assert data["store_info"]["store_id"] == 1
    assert len(data["historical_sales"]) == 30
    
    # Sunday is June 7, 2015. Check if it predicted exactly 0 sales (closed store rule)
    june_7_pred = [p for p in data["predictions"] if p["date"] == "2015-06-07"][0]
    print(f"June 7 (Sunday) Pred: {june_7_pred}")
    assert june_7_pred["open"] == 0
    assert june_7_pred["predicted_sales"] == 0.0
    
    # Check weekday prediction is positive and real (using XGBoost)
    june_1_pred = [p for p in data["predictions"] if p["date"] == "2015-06-01"][0]
    print(f"June 1 (Monday) Pred: {june_1_pred}")
    assert june_1_pred["open"] == 1
    assert june_1_pred["predicted_sales"] > 0.0
    print("Predict endpoint check passed!")

    # 6. Test Predict Endpoint with User Overrides (What-if scenario)
    print("\n--- Testing POST /predict with overrides (close Mon, open Sun) ---")
    req_body_override = {
        "store_id": 1,
        "start_date": "2015-06-01",
        "end_date": "2015-06-07",
        "overrides": [
            {
                "date": "2015-06-01", # Force closed on Monday
                "open": 0
            },
            {
                "date": "2015-06-07", # Force open on Sunday
                "open": 1,
                "promo": 1
            }
        ]
    }
    response = client.post("/predict", json=req_body_override)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    data_ov = response.json()
    
    june_1_ov = [p for p in data_ov["predictions"] if p["date"] == "2015-06-01"][0]
    june_7_ov = [p for p in data_ov["predictions"] if p["date"] == "2015-06-07"][0]
    
    print(f"Override Monday (Force Close): {june_1_ov}")
    print(f"Override Sunday (Force Open): {june_7_ov}")
    
    assert june_1_ov["open"] == 0
    assert june_1_ov["predicted_sales"] == 0.0
    
    assert june_7_ov["open"] == 1
    assert june_7_ov["predicted_sales"] > 0.0 # Open store should now use the XGBoost model
    print("Overrides check passed!")

    # 7. Test Input Validation Endpoint
    print("\n--- Testing POST /predict validation checks ---")
    # Invalid store
    bad_req_1 = {"store_id": 9999, "start_date": "2015-06-01", "end_date": "2015-06-07"}
    response = client.post("/predict", json=bad_req_1)
    print(f"Bad Request 1 (Store ID 9999) Status: {response.status_code}")
    assert response.status_code == 400 or response.status_code == 404
    
    # Invalid dates
    bad_req_2 = {"store_id": 1, "start_date": "2015-06-07", "end_date": "2015-06-01"}
    response = client.post("/predict", json=bad_req_2)
    print(f"Bad Request 2 (Start > End Date) Status: {response.status_code}")
    assert response.status_code == 400
    
    print("Validation check passed!")
    print("\nALL BACKEND API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_backend_flow()
