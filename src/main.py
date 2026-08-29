import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ForecastRequest,
    ForecastResponse,
    StoreInfo,
)

from src.forecaster import run_recursive_forecast

from src.history_loader import (
    load_store_history,
    get_dataset_metadata,
)
# LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_rolling_xgb_model.pkl"
)

FEATURE_PATH = (
    BASE_DIR
    / "models"
    / "feature_cols.pkl"
)

STORE_PATH = (
    BASE_DIR
    / "data"
    / "store.csv"
)
# GLOBAL VARIABLES
#
# IMPORTANT:
# We intentionally DO NOT keep model_data.csv loaded here.
#
# model_data.csv is loaded only for the requested store when
# a prediction request arrives.

model = None
feature_cols = None
store_df = None

median_distance = 2330.0
global_median_sales = 5430.0

dataset_metadata = None

# FASTAPI APPLICATION

app = FastAPI(
    title="Demand Forecasting & Sales Intelligence API",
    description=(
        "Backend API for running Rossmann sales predictions "
        "using an XGBoost recursive rolling forecast model."
    ),
    version="1.0.0",
)

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# STARTUP

@app.on_event("startup")
def startup_event():

    global model
    global feature_cols
    global store_df
    global median_distance
    global global_median_sales
    global dataset_metadata

    logger.info("Starting Demand Forecasting API...")

    # 1. LOAD XGBOOST MODEL

    if not MODEL_PATH.exists():

        logger.error(
            f"Model file not found: {MODEL_PATH}"
        )

    else:

        try:

            model = joblib.load(MODEL_PATH)

            logger.info(
                "Successfully loaded XGBoost model."
            )

        except Exception as e:

            logger.error(
                f"Error loading model: {e}",
                exc_info=True,
            )

    # 2. LOAD FEATURE COLUMNS
    if not FEATURE_PATH.exists():

        logger.error(
            f"Feature columns file not found: {FEATURE_PATH}"
        )

    else:

        try:

            feature_cols = joblib.load(FEATURE_PATH)

            logger.info(
                f"Loaded {len(feature_cols)} feature columns."
            )

        except Exception as e:

            logger.error(
                f"Error loading feature columns: {e}",
                exc_info=True,
            )

    # 3. LOAD STORE CONFIGURATION

    if not STORE_PATH.exists():

        logger.error(
            f"Store CSV not found: {STORE_PATH}"
        )

    else:

        try:

            store_df = pd.read_csv(STORE_PATH)

            if (
                "CompetitionDistance"
                in store_df.columns
            ):

                median_distance = float(
                    store_df[
                        "CompetitionDistance"
                    ].median()
                )

            logger.info(
                "Loaded store configuration data for "
                f"{len(store_df)} stores."
            )

            logger.info(
                f"Median competition distance: "
                f"{median_distance}"
            )

        except Exception as e:

            logger.error(
                f"Error loading store data: {e}",
                exc_info=True,
            )


    try:

        logger.info(
            "Loading lightweight historical dataset metadata..."
        )

        dataset_metadata = get_dataset_metadata()

        if dataset_metadata is not None:

            # Expected metadata values
            global_median_sales = float(
                dataset_metadata.get(
                    "global_median_sales",
                    global_median_sales,
                )
            )

            logger.info(
                "Historical dataset metadata loaded successfully."
            )

            logger.info(
                f"Global median sales: "
                f"{global_median_sales}"
            )

        else:

            logger.warning(
                "Dataset metadata could not be loaded."
            )

    except Exception as e:

        logger.error(
            f"Error loading dataset metadata: {e}",
            exc_info=True,
        )


    logger.info(
        "Demand Forecasting API startup completed."
    )

# HEALTH CHECK

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
)
def health_check():

    """
    Check whether the application and required assets
    are successfully available.
    """

    model_ok = (
        model is not None
        and feature_cols is not None
    )

    datasets_ok = (
        store_df is not None
        and dataset_metadata is not None
    )

    status = (
        "healthy"
        if model_ok and datasets_ok
        else "degraded"
    )

    return HealthResponse(
        status=status,
        model_loaded=model_ok,
        datasets_loaded=datasets_ok,
    )

# MODEL INFORMATION

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Diagnostics"],
)
def get_model_info():

    """
    Return model information and historical dataset limits.

    This endpoint uses lightweight metadata instead of loading
    the full model_data.csv dataframe.
    """

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded on server.",
        )

    if feature_cols is None:

        raise HTTPException(
            status_code=503,
            detail="Feature columns are not loaded on server.",
        )

    if dataset_metadata is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Historical dataset metadata is not "
                "available on server."
            ),
        )

    try:

        date_min = str(
            dataset_metadata["date_min"]
        )

        date_max = str(
            dataset_metadata["date_max"]
        )

        total_stores = int(
            dataset_metadata["total_stores"]
        )

    except KeyError as e:

        logger.error(
            f"Dataset metadata is missing key: {e}"
        )

        raise HTTPException(
            status_code=503,
            detail="Historical dataset metadata is incomplete.",
        )

    return ModelInfoResponse(
        features=feature_cols,
        date_min=date_min,
        date_max=date_max,
        total_stores=total_stores,
    )

# GET ALL STORES

@app.get(
    "/stores",
    response_model=List[StoreInfo],
    tags=["Stores"],
)
def get_stores_list():

    """
    Return configuration details for all stores.
    """

    if store_df is None:

        raise HTTPException(
            status_code=503,
            detail="Store dataset is not loaded.",
        )

    stores = []

    for _, row in store_df.iterrows():

        comp_dist = row.get(
            "CompetitionDistance"
        )

        stores.append(

            StoreInfo(
                store_id=int(
                    row["Store"]
                ),

                store_type=str(
                    row["StoreType"]
                ),

                assortment=str(
                    row["Assortment"]
                ),

                competition_distance=(
                    float(comp_dist)
                    if not pd.isna(comp_dist)
                    else None
                ),

                promo2=int(
                    row["Promo2"]
                ),
            )
        )

    return stores

# GET SPECIFIC STORE

@app.get(
    "/store/{store_id}",
    response_model=StoreInfo,
    tags=["Stores"],
)
def get_store_details(
    store_id: int
):

    """
    Return configuration information for one store.
    """

    if store_df is None:

        raise HTTPException(
            status_code=503,
            detail="Store dataset is not loaded.",
        )

    store_rows = store_df[
        store_df["Store"] == store_id
    ]

    if store_rows.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Store {store_id} not found."
            ),
        )

    row = store_rows.iloc[0]

    comp_dist = row.get(
        "CompetitionDistance"
    )

    return StoreInfo(

        store_id=int(
            row["Store"]
        ),

        store_type=str(
            row["StoreType"]
        ),

        assortment=str(
            row["Assortment"]
        ),

        competition_distance=(

            float(comp_dist)

            if not pd.isna(comp_dist)

            else None

        ),

        promo2=int(
            row["Promo2"]
        ),
    )

# PREDICTION ENDPOINT

@app.post(
    "/predict",
    response_model=ForecastResponse,
    tags=["Forecasting"],
)
def make_prediction(
    request: ForecastRequest
):

    """
    Generate a recursive sales forecast.

    MEMORY OPTIMISATION:

    The full model_data.csv is NOT loaded at startup.

    Instead:

        Request for Store 24
                ↓
        Load only Store 24 history
                ↓
        Pass history to recursive forecaster
                ↓
        Generate forecast
                ↓
        Request finishes
                ↓
        Temporary dataframe can be released

    This preserves the original recursive forecasting behaviour.
    """

    # CHECK MODEL

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded on server.",
        )


    if feature_cols is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Feature columns are not loaded on server."
            ),
        )

    # CHECK STORE DATA

    if store_df is None:

        raise HTTPException(
            status_code=503,
            detail="Store dataset is not loaded on server.",
        )

    # VALIDATE STORE ID

    if (
        request.store_id < 1
        or request.store_id > 1115
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Store ID must be between "
                "1 and 1115."
            ),
        )

    # VALIDATE DATES

    try:

        start_date_parsed = pd.to_datetime(
            request.start_date
        )

        end_date_parsed = pd.to_datetime(
            request.end_date
        )


        if (
            start_date_parsed
            > end_date_parsed
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Start date must be on or "
                    "before end date."
                ),
            )


        horizon_days = (
            end_date_parsed
            - start_date_parsed
        ).days + 1


        # Keep the same protection as your
        # original application.
        if horizon_days > 90:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Forecast horizon cannot "
                    "exceed 90 days."
                ),
            )


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid date format: {e}"
            ),
        )

    # LOAD ONLY THE REQUESTED STORE HISTORY

    try:

        logger.info(
            "Loading historical data for "
            f"Store {request.store_id}..."
        )


        historical_sales_df = load_store_history(
            request.store_id
        )


        if (
            historical_sales_df is None
            or historical_sales_df.empty
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No historical data found "
                    f"for Store {request.store_id}."
                ),
            )


        logger.info(
            f"Loaded {len(historical_sales_df)} "
            f"historical records for "
            f"Store {request.store_id}."
        )


    except HTTPException:

        raise


    except Exception as e:

        logger.error(
            f"Error loading store history: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load historical "
                f"data for Store {request.store_id}."
            ),
        )

    try:

        overrides_list = (

            [
                override.dict()
                for override
                in request.overrides
            ]

            if request.overrides

            else None

        )


        logger.info(
            "Running recursive forecast for "
            f"Store {request.store_id}: "
            f"{request.start_date} to "
            f"{request.end_date}"
        )


        forecast_result = run_recursive_forecast(

            model=model,

            feature_cols=feature_cols,

            store_id=request.store_id,

            start_date=request.start_date,

            end_date=request.end_date,

            overrides=overrides_list,

            store_df=store_df,

            historical_sales_df=historical_sales_df,

            median_distance=median_distance,

            global_median_sales=global_median_sales,
        )


        logger.info(
            "Forecast completed successfully "
            f"for Store {request.store_id}."
        )


        return forecast_result


    except ValueError as ve:

        logger.warning(
            f"Forecast validation error: {ve}"
        )

        raise HTTPException(
            status_code=404,
            detail=str(ve),
        )


    except Exception as e:

        logger.error(
            f"Prediction failed: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction error: {str(e)}"
            ),
        )