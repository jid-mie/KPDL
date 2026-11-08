"""Demand forecasting baselines and model comparison on chronological splits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/model'; REPORT=ROOT/'reports'; PRED=ROOT/'data/model/predictions'
TARGET='DailyDemand'
DROP={TARGET,'Date','StockCode','UnitPrice','InvoiceCount','DailyDemand_Log','DailyDemand_Sqrt','DailyDemand_Category','DailyDemand_Class','DailyDemand_IsHigh','IsOutlierDemand','ProductClusterName'}

def main():
    REPORT.mkdir(exist_ok=True); PRED.mkdir(parents=True,exist_ok=True)
    train=pd.read_parquet(DATA/'train.parquet'); val=pd.read_parquet(DATA/'val.parquet'); test=pd.read_parquet(DATA/'test.parquet')
    # Deterministic training sample can overweight rare/high-demand dates; use
    # chronologically ordered simple random sampling for comparable metrics.
    train=train.dropna(subset=['Lag_1']).copy()
    sampled=False
    if len(train)>250_000:
        train=train.sample(n=250_000,random_state=42).sort_index(); sampled=True
    features=[c for c in train.columns if c not in DROP and c in val.columns and c in test.columns]
    cat=[c for c in ['ProductCluster'] if c in features]
    numeric=[c for c in features if c not in cat]
    transformer=ColumnTransformer([
        ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric),
        ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cat),
    ],remainder='drop')
    estimators={
        'LinearRegression':LinearRegression(),
        'RandomForest':RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_features=0.8,n_jobs=-1,random_state=42),
        'HistGradientBoosting':HistGradientBoostingRegressor(loss='absolute_error',max_iter=250,max_leaf_nodes=31,l2_regularization=2.0,random_state=42),
        'TwoStageDemand':None,
    }
    # Optional XGBoost if already installed; no hard dependency.
    from xgboost import XGBRegressor
    estimators['XGBoost']=XGBRegressor(n_estimators=250,max_depth=7,learning_rate=0.05,subsample=.8,colsample_bytree=.8,objective='reg:squarederror',n_jobs=-1,random_state=42)
    results=[]; fitted={}
    # Hurdle model: first classify sale/no-sale, then estimate positive demand.
    from sklearn.ensemble import HistGradientBoostingClassifier
    hurdle_classifier=make_pipeline(transformer,HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=31,l2_regularization=2.0,random_state=42))
    hurdle_regressor=make_pipeline(transformer,HistGradientBoostingRegressor(loss='absolute_error',max_iter=250,max_leaf_nodes=31,l2_regularization=2.0,random_state=42))
    hurdle_classifier.fit(train[features],(train[TARGET]>0).astype('int8'))
    positive_train=train[train[TARGET]>0]
    hurdle_regressor.fit(positive_train[features],positive_train[TARGET])
    fitted['TwoStageDemand']=(hurdle_classifier,hurdle_regressor)
    p_sale=hurdle_classifier.predict_proba(val[features])[:,1]
    pos_hat=np.maximum(0,hurdle_regressor.predict(val[features]))
    # Tune the sale-probability cutoff on validation only. The thresholded
    # hurdle forecast is compared alongside the probabilistic expectation.
    threshold_grid=np.arange(0.10,0.91,0.05)
    threshold_scores=[]
    for threshold in threshold_grid:
        candidate=(p_sale>=threshold)*pos_hat
        threshold_scores.append({'threshold':round(float(threshold),2),'metrics':metrics(val[TARGET],candidate)})
    best_threshold=min(threshold_scores,key=lambda x:x['metrics']['MAE'])['threshold']
    hurdle_pred=(p_sale>=best_threshold)*pos_hat
    results.append({'model':'TwoStageDemand','validation':metrics(val[TARGET],hurdle_pred),'sale_probability_threshold':best_threshold,'probabilistic_expected_demand_validation':metrics(val[TARGET],p_sale*pos_hat)})
    print('TwoStageDemand',results[-1]['validation'])
    for name, estimator in estimators.items():
        if estimator is None: continue
        model=make_pipeline(transformer,estimator)
        model.fit(train[features],train[TARGET])
        fitted[name]=model
        pred=np.maximum(0,model.predict(val[features]))
        results.append({'model':name,'validation':metrics(val[TARGET],pred)})
        print(name,results[-1]['validation'])
    # Choose by validation MAE, then evaluate once on held-out test.
    best=min(results,key=lambda x:x['validation']['MAE'])['model']
    if best=='TwoStageDemand':
        clf,reg=fitted[best]; test_pred=(clf.predict_proba(test[features])[:,1]>=best_threshold)*np.maximum(0,reg.predict(test[features]))
    else:
        test_pred=np.maximum(0,fitted[best].predict(test[features]))
    test_result=metrics(test[TARGET],test_pred)
    out=test[['StockCode','Date',TARGET]].copy(); out['PredictedDemand']=test_pred
    out.to_parquet(PRED/'test_predictions.parquet',index=False)
    out.head(2000).to_csv(PRED/'test_predictions_sample.csv',index=False,encoding='utf-8-sig')
    # Aggregate a readable sample plot to avoid rendering nearly a million points.
    plot_daily=out.groupby('Date')[[TARGET,'PredictedDemand']].sum()
    fig,ax=plt.subplots(figsize=(12,5)); plot_daily.plot(ax=ax); ax.set(title=f'Daily demand: actual vs {best}',ylabel='Quantity',xlabel='Date'); fig.tight_layout(); fig.savefig(REPORT/'figures/demand_forecast_test.png',dpi=160); plt.close(fig)
    # Persist validation comparison and final test metric; baseline is last observed daily demand.
    baseline=np.maximum(0,test['Lag_1'].fillna(0).to_numpy())
    report={'target':TARGET,'feature_count':len(features),'features':features,'train_rows_used':len(train),'train_sampled':sampled,'validation_rows':len(val),'test_rows':len(test),
      'validation_metrics':results,'selected_model':best,'sale_probability_threshold':best_threshold if best=='TwoStageDemand' else None,'threshold_search_validation':threshold_scores if best=='TwoStageDemand' else [],'test_metrics':test_result,'lag1_baseline_test_metrics':metrics(test[TARGET],baseline),
      'metrics_definition':{'MAE':'mean absolute error','RMSE':'root mean squared error','WAPE':'sum absolute error / sum actual','R2':'coefficient of determination'},
      'notes':['Target-derived transformations, target class, contemporaneous price and invoice count, and outlier flags are excluded from predictors.','TwoStageDemand predicts sale probability then positive demand, with expected demand equal to probability times positive-demand estimate.','Validation selects the model; test is evaluated once after selection.','Negative direct demand forecasts are clipped to zero.','WAPE above 1 means aggregate absolute error is larger than aggregate actual demand.']}
    (REPORT/'modeling_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected_model':best,'test_metrics':test_result},indent=2))

def metrics(actual,pred):
    actual=np.asarray(actual,dtype=float); pred=np.asarray(pred,dtype=float)
    denom=np.abs(actual).sum()
    return {'MAE':round(float(mean_absolute_error(actual,pred)),5),'RMSE':round(float(np.sqrt(mean_squared_error(actual,pred))),5),'WAPE':round(float(np.abs(actual-pred).sum()/denom),5) if denom else None,'R2':round(float(r2_score(actual,pred)),5)}

if __name__=='__main__': main()
