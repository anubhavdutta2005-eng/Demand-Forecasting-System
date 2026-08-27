import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def prepare_store_features(store_row: pd.Series, median_distance: float) -> Dict[str, Any]:
    """
    Extracts and maps static store features.
    """
    store_type = str(store_row.get("StoreType", "a")).strip()
    assortment = str(store_row.get("Assortment", "a")).strip()
    
    comp_dist = store_row.get("CompetitionDistance")
    if pd.isna(comp_dist):
        comp_dist = median_distance
    
    return {
        "CompetitionDistance": float(comp_dist),
        "Promo2": int(store_row.get("Promo2", 0)),
        "StoreType_a": 1 if store_type == "a" else 0,
        "StoreType_b": 1 if store_type == "b" else 0,
        "StoreType_c": 1 if store_type == "c" else 0,
        "StoreType_d": 1 if store_type == "d" else 0,
        "Assortment_a": 1 if assortment == "a" else 0,
        "Assortment_b": 1 if assortment == "b" else 0,
        "Assortment_c": 1 if assortment == "c" else 0,
        "CompetitionOpenSinceYear": store_row.get("CompetitionOpenSinceYear"),
        "CompetitionOpenSinceMonth": store_row.get("CompetitionOpenSinceMonth"),
        "PromoInterval": store_row.get("PromoInterval"),
        "StoreType": store_type,
        "Assortment": assortment
    }

def run_recursive_forecast(
    model,
    feature_cols: List[str],
    store_id: int,
    start_date: str,
    end_date: str,
    overrides: Optional[List[Dict[str, Any]]],
    store_df: pd.DataFrame,
    historical_sales_df: pd.DataFrame,
    median_distance: float,
    global_median_sales: float
) -> Dict[str, Any]:
    """
    Runs a leak-safe recursive forecast for a single store over the specified range.
    """
    # 1. Get store details
    store_rows = store_df[store_df["Store"] == store_id]
    if store_rows.empty:
        raise ValueError(f"Store {store_id} not found in store data")
    store_row = store_rows.iloc[0]
    store_feats = prepare_store_features(store_row, median_distance)
    
    # 2. Get store historical sales series
    # Filter historical data for this store. Convert dates to datetime.
    store_history = historical_sales_df[historical_sales_df["Store"] == store_id].copy()
    store_history["Date"] = pd.to_datetime(store_history["Date"])
    store_history = store_history.sort_values("Date")
    
    # store median sales to fill NaNs if needed
    store_sales = store_history[store_history["Open"] == 1]["Sales"]
    store_median_sales = store_sales.median() if not store_sales.empty else global_median_sales
    if pd.isna(store_median_sales):
        store_median_sales = 0.0
        
    sales_history = store_history.set_index("Date")["Sales"].to_dict()
    
    # 3. Establish the forecasting dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    forecast_dates = pd.date_range(start=start_dt, end=end_dt)
    
    # Create overrides lookup dictionary
    override_dict = {}
    if overrides:
        for item in overrides:
            item_dt = pd.to_datetime(item["date"]).date()
            override_dict[item_dt] = item

    # 4. Resolve baseline schedule for the dates (if available in historical data, else default)
    # Fetch existing records in our model_data if it contains these dates
    baseline_records = store_history[
        (store_history["Date"] >= start_dt) & (store_history["Date"] <= end_dt)
    ].set_index("Date")
    
    results = []
    
    # Helper to calculate lag sales
    def get_lag_sales(date, lag_days):
        target = date - timedelta(days=lag_days)
        return sales_history.get(target, np.nan)
        
    # Helper to calculate rolling mean
    def get_rolling_mean(date, window_days):
        vals = []
        for i in range(1, window_days + 1):
            target = date - timedelta(days=i)
            val = sales_history.get(target)
            if val is not None and not pd.isna(val):
                vals.append(val)
        if not vals:
            return np.nan
        return float(np.mean(vals))

    for dt in forecast_dates:
        # Determine schedule values for this day
        # Default fallback:
        day_of_week = dt.dayofweek + 1 # 1=Mon, 7=Sun
        is_sunday = (day_of_week == 7)
        
        is_open = 0 if is_sunday else 1
        promo = 0
        school_holiday = 0
        state_holiday = "0"
        
        # Check baseline dataset for actual values if they exist
        if dt in baseline_records.index:
            row = baseline_records.loc[dt]
            # Handle possible duplicate index rows (just take first)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            is_open = int(row.get("Open", is_open))
            promo = int(row.get("Promo", promo))
            school_holiday = int(row.get("SchoolHoliday", school_holiday))
            state_holiday = str(row.get("StateHoliday", state_holiday))
            actual_sales_val = float(row.get("Sales"))
        else:
            actual_sales_val = None

        # Apply user overrides if present
        dt_date = dt.date()
        if dt_date in override_dict:
            ov = override_dict[dt_date]
            if ov.get("open") is not None:
                is_open = int(ov["open"])
            if ov.get("promo") is not None:
                promo = int(ov["promo"])
            if ov.get("school_holiday") is not None:
                school_holiday = int(ov["school_holiday"])
            if ov.get("state_holiday") is not None:
                state_holiday = str(ov["state_holiday"])

        # Derivations
        is_state_holiday = 1 if state_holiday != "0" else 0
        
        # StateHoliday one-hot features
        sh_0 = 1 if state_holiday == "0" else 0
        sh_a = 1 if state_holiday == "a" else 0
        sh_b = 1 if state_holiday == "b" else 0
        sh_c = 1 if state_holiday == "c" else 0
        
        # Competition Open Months
        comp_open_months = 0
        comp_yr = store_feats.get("CompetitionOpenSinceYear")
        comp_mo = store_feats.get("CompetitionOpenSinceMonth")
        if not pd.isna(comp_yr) and not pd.isna(comp_mo):
            try:
                comp_open_dt = pd.Timestamp(year=int(comp_yr), month=int(comp_mo), day=1)
                months = (dt.year - comp_open_dt.year) * 12 + (dt.month - comp_open_dt.month)
                comp_open_months = max(0, months)
            except Exception as e:
                logger.warning(f"Error parsing competition open date: {e}")
                pass
                
        # Promo2 Active
        promo2_active = 0
        promo2 = store_feats.get("Promo2", 0)
        promo_interval = store_feats.get("PromoInterval")
        if promo2 == 1 and not pd.isna(promo_interval):
            promo_intervals = {
                "Jan,Apr,Jul,Oct": [1, 4, 7, 10],
                "Feb,May,Aug,Nov": [2, 5, 8, 11],
                "Mar,Jun,Sept,Dec": [3, 6, 9, 12]
            }
            months = promo_intervals.get(str(promo_interval).strip(), [])
            if dt.month in months:
                promo2_active = 1
                
        # Generate lag and rolling features
        lag_1 = get_lag_sales(dt, 1)
        lag_7 = get_lag_sales(dt, 7)
        lag_14 = get_lag_sales(dt, 14)
        roll_7 = get_rolling_mean(dt, 7)
        roll_14 = get_rolling_mean(dt, 14)
        roll_30 = get_rolling_mean(dt, 30)
        
        # Fill missing values in lag/rolling features using store median sales
        # (This handles the case where historical data is sparse or forecast goes deep into the future)
        if pd.isna(lag_1): lag_1 = store_median_sales
        if pd.isna(lag_7): lag_7 = store_median_sales
        if pd.isna(lag_14): lag_14 = store_median_sales
        if pd.isna(roll_7): roll_7 = store_median_sales
        if pd.isna(roll_14): roll_14 = store_median_sales
        if pd.isna(roll_30): roll_30 = store_median_sales

        # Construct feature vector dictionary
        feat_dict = {
            "Store": int(store_id),
            "DayOfWeek": int(day_of_week),
            "Open": int(is_open),
            "Promo": int(promo),
            "SchoolHoliday": int(school_holiday),
            "CompetitionDistance": float(store_feats["CompetitionDistance"]),
            "Promo2": int(store_feats["Promo2"]),
            "Year": int(dt.year),
            "Month": int(dt.month),
            "Day": int(dt.day),
            "WeekOfYear": int(dt.isocalendar().week),
            "CompetitionOpenMonths": float(comp_open_months),
            "Promo2Active": int(promo2_active),
            "IsStateHoliday": int(is_state_holiday),
            "StoreType_a": int(store_feats["StoreType_a"]),
            "StoreType_b": int(store_feats["StoreType_b"]),
            "StoreType_c": int(store_feats["StoreType_c"]),
            "StoreType_d": int(store_feats["StoreType_d"]),
            "Assortment_a": int(store_feats["Assortment_a"]),
            "Assortment_b": int(store_feats["Assortment_b"]),
            "Assortment_c": int(store_feats["Assortment_c"]),
            "StateHoliday_0": int(sh_0),
            "StateHoliday_a": int(sh_a),
            "StateHoliday_b": int(sh_b),
            "StateHoliday_c": int(sh_c),
            "Sales_Lag_1": float(lag_1),
            "Sales_Lag_7": float(lag_7),
            "Sales_Lag_14": float(lag_14),
            "Sales_Rolling_7": float(roll_7),
            "Sales_Rolling_14": float(roll_14),
            "Sales_Rolling_30": float(roll_30)
        }

        # Predict sales
        if is_open == 0:
            pred_sales = 0.0
        else:
            # Prepare feature list in exact order
            feat_values = [feat_dict[col] for col in feature_cols]
            
            # Predict using model
            # Ensure 2D input array for prediction
            X = np.array([feat_values])
            pred_sales = float(model.predict(X)[0])
            
            # Prediction must not be negative
            pred_sales = max(0.0, pred_sales)
            
        # Update sales history with this predicted value for recursive calculations
        sales_history[dt] = pred_sales
        
        # Save results
        results.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": is_open,
            "promo": promo,
            "school_holiday": school_holiday,
            "state_holiday": state_holiday,
            "predicted_sales": round(pred_sales, 2),
            "actual_sales": actual_sales_val
        })

    # 5. Calculate metrics if we have actual sales for all prediction dates
    metrics_summary = None
    actuals = [r["actual_sales"] for r in results]
    preds = [r["predicted_sales"] for r in results]
    
    if all(a is not None for a in actuals) and len(actuals) > 0:
        actuals_arr = np.array(actuals, dtype=float)
        preds_arr = np.array(preds, dtype=float)
        
        mae = float(np.mean(np.abs(actuals_arr - preds_arr)))
        rmse = float(np.sqrt(np.mean((actuals_arr - preds_arr) ** 2)))
        sum_act = np.sum(np.abs(actuals_arr))
        wape = float(np.sum(np.abs(actuals_arr - preds_arr)) / sum_act * 100) if sum_act > 0 else 0.0
        
        # R2 score calculation
        mean_act = np.mean(actuals_arr)
        ss_res = np.sum((actuals_arr - preds_arr) ** 2)
        ss_tot = np.sum((actuals_arr - mean_act) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
        
        metrics_summary = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "wape": round(wape, 2),
            "r2": round(r2, 4)
        }
        
    # Get recent historical sales (30 days prior to start_date)
    recent_history = []
    hist_dates = pd.date_range(end=start_dt - timedelta(days=1), periods=30)
    for hdt in hist_dates:
        hsales = sales_history.get(hdt)
        if hsales is not None and not pd.isna(hsales):
            recent_history.append({
                "date": hdt.strftime("%Y-%m-%d"),
                "sales": float(hsales)
            })
            
    store_info_dict = {
        "store_id": int(store_id),
        "store_type": store_feats["StoreType"],
        "assortment": store_feats["Assortment"],
        "competition_distance": store_feats["CompetitionDistance"],
        "promo2": store_feats["Promo2"]
    }
    
    return {
        "store_id": store_id,
        "predictions": results,
        "metrics": metrics_summary,
        "store_info": store_info_dict,
        "historical_sales": recent_history
    }
