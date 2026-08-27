from pydantic import BaseModel, Field
from typing import List, Optional

class DayScheduleOverride(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    open: Optional[int] = Field(None, description="1 if open, 0 if closed")
    promo: Optional[int] = Field(None, description="1 if promotion active, 0 if not")
    school_holiday: Optional[int] = Field(None, description="1 if school holiday, 0 if not")
    state_holiday: Optional[str] = Field(None, description="Holiday code: '0', 'a', 'b', 'c'")

class ForecastRequest(BaseModel):
    store_id: int = Field(..., description="Store ID (1 to 1115)")
    start_date: str = Field(..., description="Start date of forecast (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date of forecast (YYYY-MM-DD)")
    overrides: Optional[List[DayScheduleOverride]] = Field(None, description="List of custom overrides for specific dates")

class ForecastItem(BaseModel):
    date: str
    open: int
    promo: int
    school_holiday: int
    state_holiday: str
    predicted_sales: float
    actual_sales: Optional[float] = None

class MetricSummary(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    wape: Optional[float] = None
    r2: Optional[float] = None

class StoreInfo(BaseModel):
    store_id: int
    store_type: str
    assortment: str
    competition_distance: Optional[float] = None
    promo2: int

class HistoricalSalesItem(BaseModel):
    date: str
    sales: float

class ForecastResponse(BaseModel):
    store_id: int
    predictions: List[ForecastItem]
    metrics: Optional[MetricSummary] = None
    store_info: StoreInfo
    historical_sales: List[HistoricalSalesItem]

class ModelInfoResponse(BaseModel):
    features: List[str]
    date_min: str
    date_max: str
    total_stores: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    datasets_loaded: bool
