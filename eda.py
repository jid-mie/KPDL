"""Create reproducible descriptive tables and plots for Online Retail II."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; REPORT=ROOT/'reports'; FIG=ROOT/'reports/figures'
def main():
 REPORT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
 tx=pd.read_parquet(ROOT/'data/processed/transactions_clean.parquet'); tx['Date']=pd.to_datetime(tx.Date)
 daily=tx.groupby('Date').agg(Quantity=('Quantity','sum'),Revenue=('Revenue','sum'),Invoices=('InvoiceNo','nunique')).sort_index()
 product=tx.groupby('StockCode').agg(Transactions=('InvoiceNo','nunique'),Quantity=('Quantity','sum'),Revenue=('Revenue','sum')).sort_values('Revenue',ascending=False)
 identified=tx.dropna(subset=['CustomerID']).copy()
 customer=(identified.groupby('CustomerID').agg(InvoiceCount=('InvoiceNo','nunique'),Quantity=('Quantity','sum'),Revenue=('Revenue','sum'),FirstPurchase=('Date','min'),LastPurchase=('Date','max')).sort_values('Revenue',ascending=False))
 country=(tx.groupby('Country').agg(Transactions=('InvoiceNo','nunique'),Customers=('CustomerID','nunique'),Quantity=('Quantity','sum'),Revenue=('Revenue','sum')).sort_values('Revenue',ascending=False))
 country['RevenueShare']=country.Revenue/country.Revenue.sum()
 monthly=daily.resample('MS').sum()
 weekly=daily.resample('W-MON').sum()
 topn=int(max(1,round(len(product)*.1)))
 report={'rows':len(tx),'products':int(tx.StockCode.nunique()),'customers':int(tx.CustomerID.nunique()),'countries':int(tx.Country.nunique()),'date_range':[tx.Date.min().strftime('%Y-%m-%d'),tx.Date.max().strftime('%Y-%m-%d')],
 'quantity_distribution':{k:float(v) for k,v in tx.Quantity.describe(percentiles=[.5,.9,.95,.99]).to_dict().items()},
 'unit_price_distribution':{k:float(v) for k,v in tx.UnitPrice.describe(percentiles=[.5,.9,.95,.99]).to_dict().items()},
 'top_10_products_by_revenue':product.head(10).reset_index().to_dict('records'),
 'top_10pct_products_revenue_share':float(product.head(topn).Revenue.sum()/product.Revenue.sum()),
 'customer_id_missing_pct':float(tx.CustomerID.isna().mean()),'identified_customer_count':int(len(customer)),
 'customer_revenue_quantiles':{k:float(v) for k,v in customer.Revenue.quantile([0,.25,.5,.75,.9,.99,1]).items()},
 'top_10_customers_by_revenue':customer.head(10).reset_index().to_dict('records'),
 'country_top_10_by_revenue':country.head(10).reset_index().to_dict('records'),
 'country_revenue_concentration_top_3':float(country.head(3).RevenueShare.sum()),
 'monthly_rows':len(monthly),
 'weekly_rows':len(weekly),'monthly_quantity_peak':{'period':monthly.Quantity.idxmax().strftime('%Y-%m'),'quantity':int(monthly.Quantity.max())},
 'monthly_revenue_peak':{'period':monthly.Revenue.idxmax().strftime('%Y-%m'),'revenue':float(monthly.Revenue.max())}}
 (REPORT/'eda_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 monthly.to_csv(REPORT/'monthly_sales.csv',encoding='utf-8-sig')
 weekly.to_csv(REPORT/'weekly_sales.csv',encoding='utf-8-sig')
 product.to_csv(REPORT/'product_summary.csv',encoding='utf-8-sig')
 customer.to_csv(REPORT/'customer_summary.csv',encoding='utf-8-sig')
 country.to_csv(REPORT/'country_summary.csv',encoding='utf-8-sig')
 fig,ax=plt.subplots(figsize=(12,5)); monthly.Quantity.plot(ax=ax); ax.set(title='Monthly quantity sold',ylabel='Quantity',xlabel='Month'); fig.tight_layout(); fig.savefig(FIG/'monthly_quantity.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,5)); monthly.Revenue.plot(ax=ax,color='darkorange'); ax.set(title='Monthly revenue',ylabel='Revenue',xlabel='Month'); fig.tight_layout(); fig.savefig(FIG/'monthly_revenue.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,5)); weekly.Quantity.plot(ax=ax,color='seagreen'); ax.set(title='Weekly quantity sold',ylabel='Quantity',xlabel='Week'); fig.tight_layout(); fig.savefig(FIG/'weekly_quantity.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,6)); country.head(10).sort_values('Revenue').Revenue.plot.barh(ax=ax,color='slateblue'); ax.set(title='Top 10 countries by revenue',xlabel='Revenue',ylabel='Country'); fig.tight_layout(); fig.savefig(FIG/'country_revenue_top10.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,5)); customer.Revenue.clip(upper=customer.Revenue.quantile(.99)).hist(bins=50,ax=ax); ax.set(title='Customer revenue distribution (clipped at P99)',xlabel='Revenue per identified customer',ylabel='Customers'); fig.tight_layout(); fig.savefig(FIG/'customer_revenue_distribution.png',dpi=160); plt.close(fig)
 print(json.dumps({k:report[k] for k in ['rows','products','customers','countries','top_10pct_products_revenue_share']} ,indent=2))
if __name__=='__main__': main()
