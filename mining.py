"""Association-rule mining and daily demand/revenue anomaly detection."""
from __future__ import annotations
import json
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / 'data/processed/transactions_clean.parquet'
DAILY = ROOT / 'data/processed/product_daily_demand.parquet'
OUT = ROOT / 'reports'


def apriori_rules(transactions: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.20, max_len: int = 3):
    """Small, deterministic Apriori implementation; transactions are sets of items."""
    baskets = [set(g['StockCode'].astype(str)) for _, g in transactions.groupby('InvoiceNo', sort=False)]
    n = len(baskets)
    if not n:
        return []
    support = {}
    singles = {}
    for basket in baskets:
        for item in basket:
            singles[item] = singles.get(item, 0) + 1
    current = {frozenset([i]): c / n for i, c in singles.items() if c / n >= min_support}
    support.update(current)
    k = 2
    while current and k <= max_len:
        prev = list(current)
        candidates = set()
        for a, b in combinations(prev, 2):
            union = a | b
            if len(union) == k and all(frozenset(s) in support for s in combinations(union, k-1)):
                candidates.add(union)
        counts = {c: 0 for c in candidates}
        # Index only frequent singleton products; enumerate pairs inside each
        # basket instead of scanning every candidate for every invoice.
        if k == 2:
            frequent_items = {next(iter(item)) for item in current}
            for basket in baskets:
                present = sorted(basket & frequent_items)
                for pair in combinations(present, 2):
                    candidate = frozenset(pair)
                    if candidate in counts:
                        counts[candidate] += 1
        else:
            for basket in baskets:
                present = sorted(basket)
                for group in combinations(present, k):
                    candidate = frozenset(group)
                    if candidate in counts:
                        counts[candidate] += 1
        current = {c: count / n for c, count in counts.items() if count / n >= min_support}
        support.update(current)
        k += 1
    rules = []
    for itemset, sup in support.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for left_tuple in combinations(itemset, size):
                left = frozenset(left_tuple)
                right = itemset - left
                confidence = sup / support.get(left, 0)
                if confidence >= min_confidence:
                    consequent_support = support.get(right, 0)
                    rules.append({
                        'antecedent': sorted(left), 'consequent': sorted(right),
                        'support': round(sup, 6), 'confidence': round(confidence, 6),
                        'lift': round(confidence / consequent_support, 4) if consequent_support else None,
                    })
    return sorted(rules, key=lambda r: (r['lift'] or 0, r['confidence'], r['support']), reverse=True)


def detect_anomalies(clean: pd.DataFrame):
    tx = clean.copy()
    tx['Date'] = pd.to_datetime(tx['Date'])
    daily = tx.groupby('Date').agg(TotalQuantity=('Quantity', 'sum'), TotalRevenue=('Revenue', 'sum'), InvoiceCount=('InvoiceNo', 'nunique')).reset_index()
    daily = daily.sort_values('Date')
    daily['QuantityRollingMedian28'] = daily['TotalQuantity'].rolling(28, min_periods=7).median().shift(1)
    daily['RevenueRollingMedian28'] = daily['TotalRevenue'].rolling(28, min_periods=7).median().shift(1)
    for col, baseline, flag in [('TotalQuantity','QuantityRollingMedian28','QuantityAnomaly'),('TotalRevenue','RevenueRollingMedian28','RevenueAnomaly')]:
        ratio = daily[col] / daily[baseline].replace(0, np.nan)
        daily[flag] = ((ratio >= 3.0) | (ratio <= 0.33)).fillna(False)
    # Isolation Forest on robust log-scaled aggregate signals, fitted to the full historical period for discovery only.
    x = np.log1p(daily[['TotalQuantity','TotalRevenue','InvoiceCount']].clip(lower=0))
    labels = IsolationForest(contamination=0.02, random_state=42).fit_predict(x)
    daily['IsolationForestAnomaly'] = labels == -1
    # Also score per-product daily demand, matching the project's demand-level
    # anomaly objective. Keep these rows as flags; do not silently drop them.
    product = tx[tx['StockCode'].astype(str).str.contains(r'\d', regex=True, na=False)]
    product_daily = product.groupby(['StockCode','Date'], as_index=False).agg(DailyDemand=('Quantity','sum'))
    # Do not flag ordinary no-sale days. Evaluate unusual *positive* demand
    # relative to the product's previous positive-demand distribution.
    def _robust_positive_flags(s):
        positive=s[s>0]
        q1=positive.shift(1).rolling(90,min_periods=12).quantile(.25)
        q3=positive.shift(1).rolling(90,min_periods=12).quantile(.75)
        iqr=q3-q1
        flags=(positive>q3+1.5*iqr).reindex(s.index,fill_value=False)
        return flags.astype(bool)
    product_daily['DemandAnomaly'] = product_daily.groupby('StockCode')['DailyDemand'].transform(_robust_positive_flags).fillna(False)
    product_daily['AnomalyMethod']='positive demand; trailing 90 calendar observations; upper Tukey fence Q3+1.5*IQR; minimum 12 prior positive observations'
    product_daily.to_parquet(OUT/'product_demand_anomalies.parquet',index=False)
    daily.attrs['product_demand_anomalies'] = int(product_daily.DemandAnomaly.sum())
    daily.attrs['product_demand_anomaly_frame'] = product_daily
    return daily


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tx = pd.read_parquet(CLEAN)
    # Frequent itemsets can be expensive; focus association analysis on the dominant UK market.
    tx = tx[~tx['InvoiceNo'].astype(str).str.startswith('C')]
    baskets_tx = tx[tx['StockCode'].astype(str).str.contains(r'\d', regex=True, na=False)]
    rules = apriori_rules(baskets_tx, min_support=0.005, min_confidence=0.15, max_len=2)
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df.to_csv(OUT/'association_rules.csv',index=False,encoding='utf-8-sig')
    # Compact explanatory examples: strongest lift, strongest confidence, and
    # a support-oriented slice. Exclude tiny-support rules from interpretation.
    interpretable=[r for r in rules if r['support'] >= 0.01 and r['lift'] is not None]
    top_lift=sorted(interpretable,key=lambda r:r['lift'],reverse=True)[:10]
    top_conf=sorted(interpretable,key=lambda r:r['confidence'],reverse=True)[:10]
    top_support=sorted(interpretable,key=lambda r:r['support'],reverse=True)[:10]
    report={
        'method':'Apriori', 'invoice_count':int(baskets_tx['InvoiceNo'].nunique()),
        'min_support':0.005, 'min_confidence':0.15, 'max_itemset_length':2,
        'candidate_scope':'product-coded StockCode values in all retained transactions',
        'rules':rules, 'rule_count':len(rules), 'interpretable_rule_count_support_at_least_1pct':len(interpretable),
        'top_by_lift':top_lift,'top_by_confidence':top_conf,'top_by_support':top_support,
        'notes':['Full rules are also exported to association_rules.csv.', 'For narrative interpretation, rules below 1% support are excluded from the ranked examples.', 'Association does not establish causation.']
    }
    (OUT/'association_rules.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if interpretable:
        lines=['# Luật kết hợp Apriori — diễn giải','',f"Tập gồm {report['invoice_count']:,} hóa đơn; ngưỡng support ≥ {report['min_support']:.1%}, confidence ≥ {report['min_confidence']:.0%}. Có {len(rules):,} luật ở ngưỡng khai phá và {len(interpretable):,} luật có support ít nhất 1% được ưu tiên diễn giải.",'', '## Các luật nổi bật theo lift','', '| Nếu hóa đơn có | Thì thường có | Support | Confidence | Lift |','|---|---|---:|---:|---:|']
        for r in top_lift:
            lines.append(f"| {', '.join(r['antecedent'])} | {', '.join(r['consequent'])} | {r['support']:.2%} | {r['confidence']:.2%} | {r['lift']:.2f} |")
        lines += ['', '## Cách đọc', '', 'Support là tỷ lệ hóa đơn chứa cả hai phía của luật. Confidence là xác suất thấy sản phẩm bên phải trong các hóa đơn có sản phẩm bên trái. Lift so sánh confidence với tỷ lệ nền của sản phẩm bên phải; lift lớn hơn 1 cho thấy hai sản phẩm đồng xuất hiện thường hơn mức ngẫu nhiên độc lập. Đây là quan hệ đồng mua, không chứng minh một sản phẩm gây ra việc mua sản phẩm kia.', '', '## Lưu ý phương pháp', '', 'Các luật được xếp hạng cần được kiểm tra cùng support, vì lift có thể rất cao khi sự kiện hiếm. Kết quả phục vụ khám phá giỏ hàng; trước khi dùng cho gợi ý bán hàng cần đánh giá ổn định theo thời gian và kiểm tra mô tả sản phẩm.']
        (OUT/'association_interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    anomalies = detect_anomalies(tx)
    anomalies.to_csv(OUT/'daily_anomalies.csv', index=False, encoding='utf-8-sig')
    product_flags=anomalies.attrs['product_demand_anomaly_frame']
    product_anomaly_count=int(product_flags.DemandAnomaly.sum())
    product_positive_count=int((product_flags.DailyDemand>0).sum())
    # Avoid carrying a DataFrame in attrs through pandas ranking operations.
    anomalies.attrs = {}
    report = {
        'days':len(anomalies),
        'quantity_rule_anomalies':int(anomalies['QuantityAnomaly'].sum()),
        'revenue_rule_anomalies':int(anomalies['RevenueAnomaly'].sum()),
        'isolation_forest_anomalies':int(anomalies['IsolationForestAnomaly'].sum()),
        'product_demand_anomalies':product_anomaly_count,
        'product_demand_positive_rows':product_positive_count,
        'product_demand_anomaly_pct_of_positive_days':round(float(product_anomaly_count/max(1,product_positive_count)*100),3),
        'top_quantity_days':anomalies.nlargest(10,'TotalQuantity')[['Date','TotalQuantity','TotalRevenue']].assign(Date=lambda d:d.Date.dt.strftime('%Y-%m-%d')).to_dict('records'),
        'threshold':'Aggregate series: trailing 28-day median, shifted by one day; ratio >=3 or <=0.33. Product series: only positive demand can be flagged, trailing 90 calendar observations, upper fence Q3+1.5*IQR, at least 12 prior positive observations. IsolationForest contamination=2% is exploratory.'
    }
    (OUT/'anomaly_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'rules':len(rules),'anomaly_days':report['isolation_forest_anomalies']},indent=2))

if __name__ == '__main__': main()
