export type Feature = {
    Area: string;
    Item: string;
    Year: number;
    average_rain_fall_mm_per_year: number;
    pesticides_tonnes: number;
    avg_temp: number;
};
export type PredictValue = {
    hg_ha_yield: number;
};
export type Metric = {
    MSE?: number;
    MAE?: number;
    RMSE?: number;
    R2?: number;
    avgDiffToGroundTruth?: number;
};

export type LinearTrain = {
    fit_intercept: boolean;
};
