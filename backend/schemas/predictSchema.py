from pydantic import BaseModel
from typing import Optional, List

class PredictRequest(BaseModel):
    Area: str
    Item: str
    Year: int
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

class PredictResponse(BaseModel):
    hg_ha_yield: float

class DecisionTreeParam(BaseModel):
    min_samples_split: Optional[int] = None
    max_depth: Optional[int] = None
    min_samples_leaf: Optional[int] = None
    random_state: Optional[int] = None

class AnnParam(BaseModel):
    layer_sizes: Optional[List[int]] = None
    learning_rate: Optional[float] = None
    max_iter: Optional[int] = None
    verbose: Optional[bool] = None

class LassoParam(BaseModel):
    alpha: Optional[float] = None
    max_iter: Optional[int] = None
    fit_intercept: Optional[bool] = None

class MetricsResponse(BaseModel):
    MSE: float
    R2: float
