import os
import logging
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from src.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ForecastRequest,
    ForecastResponse,
    StoreInfo
)
from src.forecaster import run_recursive_forecast

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_rolling_xgb_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_cols.pkl"
STORE_PATH = BASE_DIR / "data" / "store.csv"
MODEL_DATA_PATH = BASE_DIR / "data" / "processed" / "model_data.csv"

# Globals
model = None
feature_cols = None
store_df = None
model_data_df = None
median_distance = 2330.0  # Rossmann dataset median competition distance
global_median_sales = 0.0

app = FastAPI(
    title="Demand Forecasting & Sales Intelligence API",
    description="Backend API for running Rossmann sales predictions using an XGBoost rolling forecast model.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    global model, feature_cols, store_df, model_data_df, median_distance, global_median_sales
    
    logger.info("Starting up Demand Forecasting API...")
    
    # 1. Load Model
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded XGBoost model.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
    # 2. Load Feature Columns
    if not FEATURE_PATH.exists():
        logger.error(f"Feature columns pickle not found at {FEATURE_PATH}")
    else:
        try:
            feature_cols = joblib.load(FEATURE_PATH)
            logger.info(f"Loaded {len(feature_cols)} feature columns.")
        except Exception as e:
            logger.error(f"Error loading feature columns: {e}")
            
    # 3. Load Store Data
    if not STORE_PATH.exists():
        logger.error(f"Store details CSV not found at {STORE_PATH}")
    else:
        try:
            store_df = pd.read_csv(STORE_PATH)
            median_distance = float(store_df["CompetitionDistance"].median())
            logger.info(f"Loaded store configuration data for {len(store_df)} stores. Median distance: {median_distance}")
        except Exception as e:
            logger.error(f"Error loading store data: {e}")
            
    # 4. Load Preprocessed Model Data (Historical Sales reference)
    if not MODEL_DATA_PATH.exists():
        logger.error(f"Model data CSV not found at {MODEL_DATA_PATH}")
    else:
        try:
            logger.info("Loading historical sales reference dataset (this might take a few seconds)...")
            model_data_df = pd.read_csv(MODEL_DATA_PATH)
            
            # Compute global median sales for fallback purposes
            open_sales = model_data_df[model_data_df["Open"] == 1]["Sales"]
            global_median_sales = float(open_sales.median()) if not open_sales.empty else 5430.0
            
            logger.info(f"Loaded historical dataset with {len(model_data_df)} records. Global median sales: {global_median_sales}")
        except Exception as e:
            logger.error(f"Error loading historical dataset: {e}")

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """
    Check server and asset loading health.
    """
    model_ok = model is not None
    datasets_ok = (store_df is not None) and (model_data_df is not None)
    status = "healthy" if (model_ok and datasets_ok) else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_ok,
        datasets_loaded=datasets_ok
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Diagnostics"])
def get_model_info():
    """
    Get information about the loaded model and training data limits.
    """
    if model is None or feature_cols is None or model_data_df is None:
        raise HTTPException(status_code=503, detail="Model assets or datasets are not loaded on server.")
        
    # Get date range of historical data
    model_data_df["Date"] = pd.to_datetime(model_data_df["Date"])
    date_min = model_data_df["Date"].min().strftime("%Y-%m-%d")
    date_max = model_data_df["Date"].max().strftime("%Y-%m-%d")
    total_stores = int(model_data_df["Store"].nunique())
    
    return ModelInfoResponse(
        features=feature_cols,
        date_min=date_min,
        date_max=date_max,
        total_stores=total_stores
    )

@app.get("/stores", response_model=List[StoreInfo], tags=["Stores"])
def get_stores_list():
    """
    Get a list of all store configurations.
    """
    if store_df is None:
        raise HTTPException(status_code=503, detail="Store dataset is not loaded.")
        
    stores = []
    for _, row in store_df.iterrows():
        comp_dist = row.get("CompetitionDistance")
        stores.append(
            StoreInfo(
                store_id=int(row["Store"]),
                store_type=str(row["StoreType"]),
                assortment=str(row["Assortment"]),
                competition_distance=float(comp_dist) if not pd.isna(comp_dist) else None,
                promo2=int(row["Promo2"])
            )
        )
    return stores

@app.get("/store/{store_id}", response_model=StoreInfo, tags=["Stores"])
def get_store_details(store_id: int):
    """
    Get configuration details for a specific store.
    """
    if store_df is None:
        raise HTTPException(status_code=503, detail="Store dataset is not loaded.")
        
    store_rows = store_df[store_df["Store"] == store_id]
    if store_rows.empty:
        raise HTTPException(status_code=404, detail=f"Store {store_id} not found.")
        
    row = store_rows.iloc[0]
    comp_dist = row.get("CompetitionDistance")
    return StoreInfo(
        store_id=int(row["Store"]),
        store_type=str(row["StoreType"]),
        assortment=str(row["Assortment"]),
        competition_distance=float(comp_dist) if not pd.isna(comp_dist) else None,
        promo2=int(row["Promo2"])
    )

@app.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
def make_prediction(request: ForecastRequest):
    """
    Generate a recursive demand forecast for a store over a given period, optionally applying schedule overrides.
    """
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on server.")
        
    if store_df is None or model_data_df is None:
        raise HTTPException(status_code=503, detail="Datasets are not loaded on server.")
        
    if request.store_id < 1 or request.store_id > 1115:
        raise HTTPException(status_code=400, detail="Store ID must be between 1 and 1115.")
        
    try:
        # Validate dates
        start_date_parsed = pd.to_datetime(request.start_date)
        end_date_parsed = pd.to_datetime(request.end_date)
        
        if start_date_parsed > end_date_parsed:
            raise HTTPException(status_code=400, detail="Start date must be on or before end date.")
            
        # Max horizon check to prevent server timeout (e.g. 90 days)
        horizon_days = (end_date_parsed - start_date_parsed).days + 1
        if horizon_days > 90:
            raise HTTPException(status_code=400, detail="Forecast horizon cannot exceed 90 days.")
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    try:
        # Run recursive forecast
        overrides_list = [o.dict() for o in request.overrides] if request.overrides else None
        
        forecast_result = run_recursive_forecast(
            model=model,
            feature_cols=feature_cols,
            store_id=request.store_id,
            start_date=request.start_date,
            end_date=request.end_date,
            overrides=overrides_list,
            store_df=store_df,
            historical_sales_df=model_data_df,
            median_distance=median_distance,
            global_median_sales=global_median_sales
        )
        return forecast_result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
