import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==========================================
# 1. 核心计算引擎
# ==========================================

class MacroAnalyzer:
    def __init__(self):
        self.window_long = 252 
        self.min_data_points = int(self.window_long * 0.85)
        # 阈值配置
        self.thresholds = {"extreme": 2.2, "high": 1.2, "low": -1.0}
    
    def fetch_all_data(self, tickers, period="5y"):
        unique_tickers = list(set(tickers))
        print(f"🚀 正在批量获取 {len(unique_tickers)} 个标的数据...")
        try:
            df = yf.download(unique_tickers, period=period, group_by='ticker', auto_adjust=False, threads=True)
            return df
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return pd.DataFrame()

    def extract_series(self, df_batch, ticker):
        try:
            if ticker not in df_batch.columns.levels[0]: return pd.Series(dtype=float)
            if 'Close' in df_batch[ticker]: s = df_batch[ticker]['Close']
            elif 'Adj Close' in df_batch[ticker]: s = df_batch[ticker]['Adj Close']
            else: return pd.Series(dtype=float)
            s = s.replace(0, np.nan).dropna()
            if s.index.tz: s.index = s.index.tz_localize(None)
            return s
        except: return pd.Series(dtype=float)

    def align_time_series(self, series1, series2):
        all_dates = series1.index.union(series2.index).sort_values()
        s1 = series1.reindex(all_dates).ffill()
        s2 = series2.reindex(all_dates).ffill()
        valid_mask = ~(s1.isna() | s2.isna())
        return s1[valid_mask], s2[valid_mask]

    def calculate_robust_z_score(self, series):
        """只负责计算原始 Z-Score，不负责判断方向风险"""
        if len(series) < self.min_data_points: return 0, 0.0, 1
        
        rolling_mean = series.rolling(window=self.window_long, min_periods=self.min_data_points).mean()
        bias_series = (series / rolling_mean) - 1
        bias_series = bias_series.dropna()
        
        if len(bias_series) < self.window_long: return 0, bias_series.iloc[-1] if not bias_series.empty else 0, 1

        bias_mean = bias_series.rolling(window=self.window_long).mean()
        bias_std = bias_series.rolling(window=self.window_long).std()
        cur_val = bias_series.iloc[-1]
        cur_std = bias_std.iloc[-1]
        
        if pd.isna(cur_std) or cur_std == 0: return 0, cur_val, 1
        
        # 原始 Z 分数：正数代表高于均线，负数代表低于均线
        z_score = (cur_val - bias_mean.iloc[-1]) / cur_std
        z_score = np.clip(z_score, -4.5, 4.5)
        
        return z_score, cur_val, 0

    def get_status_text(self, z, inverse):
        """
        【核心修正】逻辑拆分
        inverse=False (VIX): Z越小越好 (Green), Z越大越危险 (Red)
        inverse=True (BTC):  Z越大越好 (Green), Z越小越危险 (Red)
        """
        # 1. 正常逻辑 (涨是风险：VIX, 拥挤度)
        if not inverse:
            if z > self.thresholds['extreme']: return "red", "极度恐慌/过热"
            if z > self.thresholds['high']:    return "orange", "风险积聚"
            if z < self.thresholds['low']:     return "green", "低位平稳" # 低于均线
            return "yellow", "正常震荡"

        # 2. 反向逻辑 (跌是风险：BTC, SMH, HYG)
        else:
            if z < -self.thresholds['extreme']: return "red", "严重崩盘/枯竭"
            if z < -self.thresholds['high']:    return "orange", "显著回调"
            if z > 1.0:                         return "green", "趋势强劲" # 【修正】高于均线是好事
            return "yellow", "正常震荡"

    def analyze_indicator(self, name, desc, risk_rule, series, inverse=False, display_ticker=""):
        try:
            if series.empty: raise ValueError("无数据")
            z_score, bias, status = self.calculate_robust_z_score(series)
            current_val = series.iloc[-1]

            if status == 1:
                return {"name": name, "value": current_val, "bias": bias, "z": 0, "level": "gray", "text": "数据不足", "desc": desc, "risk_rule": risk_rule}

            # 获取修正后的评语
            level, text = self.get_status_text(z_score, inverse)
            
            return {
                "name": name, "value": current_val, "bias": bias,
                "z": z_score, "level": level, "text": text, 
                "desc": desc, "risk_rule": risk_rule, "ticker": display_ticker
            }
        except Exception as e:
            return {"name": name, "value": 0, "level": "gray", "text": "Error", "desc": desc, "risk_rule": "数据错误"}

# ==========================================
# 2. 指标配置 (文案逻辑补全)
# ==========================================

def get_us_indicators_optimized():
    analyzer = MacroAnalyzer()
    tickers_config = {
        "market": ["^VIX", "VIXY", "XLY", "XLP", "RSP", "SPY", "QQQ"],
        "credit": ["HYG", "JNK"],
        "rates": ["^TNX", "^IRX"], 
        "crypto": ["BTC-USD"],
        "tech": ["SMH"]
    }
    all_tickers = [t for sublist in tickers_config.values() for t in sublist]
    df_batch = analyzer.fetch_all_data(all_tickers)
    
    indicators = {"E (预期 Sentiment)": [], "S (结构 Structure)": [], "P (权力 Power)": [], "T (技术 Tech)": []}

    # --- E: 预期 ---
    s_vix = analyzer.extract_series(df_batch, "^VIX")
    if s_vix.empty: s_vix = analyzer.extract_series(df_batch, "VIXY")
    indicators["E (预期 Sentiment)"].append(analyzer.analyze_indicator(
        name="恐慌指数 (VIX)", 
        desc="衡量华尔街的恐惧程度。",
        risk_rule="🔴 风险：Z > 2.0 代表极度恐慌。<br>🟢 安全：Z < -1.0 代表情绪平稳。",
        series=s_vix, inverse=False # 涨是风险
    ))
    
    s_xly = analyzer.extract_series(df_batch, "XLY")
    s_xlp = analyzer.extract_series(df_batch, "XLP")
    if not s_xly.empty and not s_xlp.empty:
        s_xly, s_xlp = analyzer.align_time_series(s_xly, s_xlp)
        indicators["E (预期 Sentiment)"].append(analyzer.analyze_indicator(
            name="贪婪/防御比 (XLY/XLP)", 
            desc="资金是在进攻(消费)还是防守(必需品)？",
            risk_rule="🔴 风险：Z < -2.0 代表资金疯狂防御。<br>🟢 强劲：Z > 1.0 代表风险偏好极高。",
            series=s_xly/s_xlp, inverse=True # 跌是风险
        ))

    # --- S: 结构 ---
    s_rsp = analyzer.extract_series(df_batch, "RSP")
    s_spy = analyzer.extract_series(df_batch, "SPY")
    if not s_rsp.empty and not s_spy.empty:
        s_rsp, s_spy = analyzer.align_time_series(s_rsp, s_spy)
        indicators["S (结构 Structure)"].append(analyzer.analyze_indicator(
            name="市场广度 (RSP/SPY)", 
            desc="中小票表现 vs 巨头表现。",
            risk_rule="🔴 风险：Z < -2.0 代表由于巨头吸血，市场脆弱。<br>🟢 强劲：Z > 1.0 代表普涨牛市。",
            series=s_rsp/s_spy, inverse=True # 跌是风险
        ))

    s_hyg = analyzer.extract_series(df_batch, "HYG")
    if s_hyg.empty: s_hyg = analyzer.extract_series(df_batch, "JNK")
    indicators["S (结构 Structure)"].append(analyzer.analyze_indicator(
        name="信用底座 (HYG)", 
        desc="垃圾债价格，企业融资环境的晴雨表。",
        risk_rule="🔴 风险：Z < -2.0 代表信贷危机/借不到钱。<br>🟢 强劲：Z > 1.0 代表资金泛滥。",
        series=s_hyg, inverse=True # 跌是风险
    ))

    # --- P: 权力 ---
    s_10y = analyzer.extract_series(df_batch, "^TNX")
    s_3m = analyzer.extract_series(df_batch, "^IRX")
    if not s_10y.empty and not s_3m.empty:
        s_10y, s_3m = analyzer.align_time_series(s_10y, s_3m)
        spread = s_10y - s_3m
        indicators["P (权力 Power)"].append(analyzer.analyze_indicator(
            name="收益率曲线 (10Y-3M)", 
            desc="最准的衰退预警指标。负值即为倒挂。",
            risk_rule="🔴 风险：Z < -2.0 (深跌) 代表倒挂加剧，衰退逼近。<br>🟢 修复：Z > 1.0 代表曲线陡峭化修复。",
            series=spread, inverse=True # 低是风险
        ))
    
    s_btc = analyzer.extract_series(df_batch, "BTC-USD")
    indicators["P (权力 Power)"].append(analyzer.analyze_indicator(
        name="边际流动性 (BTC)", 
        desc="美元流动性的敏感情绪指标。",
        risk_rule="🔴 风险：Z < -2.0 代表流动性枯竭/崩盘。<br>🟢 强劲：Z > 2.0 代表流动性极度充裕。",
        series=s_btc, inverse=True # 跌是风险
    ))

    # --- T: 技术 ---
    s_qqq = analyzer.extract_series(df_batch, "QQQ")
    if not s_qqq.empty and not s_spy.empty:
        s_qqq, s_spy = analyzer.align_time_series(s_qqq, s_spy)
        indicators["T (技术 Tech)"].append(analyzer.analyze_indicator(
            name="科技拥挤度 (QQQ/SPY)", 
            desc="科技股是否过度拥挤？",
            risk_rule="🔴 风险：Z > 2.0 代表交易过热，容易踩踏。<br>🟢 安全：Z < -1.0 代表科技股无人问津。",
            series=s_qqq/s_spy, inverse=False # 涨是风险
        ))

    s_smh = analyzer.extract_series(df_batch, "SMH")
    indicators["T (技术 Tech)"].append(analyzer.analyze_indicator(
        name="AI引擎 (SMH)", 
        desc="半导体周期，本轮牛市发动机。",
        risk_rule="🔴 风险：Z < -2.0 代表牛市逻辑熄火。<br>🟢 强劲：Z > 2.0 代表AI泡沫/主升浪。",
        series=s_smh, inverse=True # 跌是风险
    ))

    return indicators

# ==========================================
# 3. 报告生成 (UI微调)
# ==========================================

def generate_html_report(indicators):
    # 熔断逻辑
    st = {}
    for cat in indicators.values():
        for item in cat:
            if "VIX" in item['name']: st['VIX'] = item['level']
            if "HYG" in item['name']: st['Credit'] = item['level']
            if "收益率" in item['name']: st['Curve'] = item['level']
            if "SMH" in item['name']: st['AI'] = item['level']
    
    overall_status = "🟢 趋势健康 (Healthy Trend)"
    summary_text = "核心指标都在正常波动范围内，没有发现明显的系统性风险。"
    bg_color = "#f0f2f5"
    header_color = "#2c3e50"
    
    veto_msgs = []
    if st.get('VIX') == 'red' and st.get('Credit') == 'red':
        veto_msgs.append("流动性危机 (VIX飙升 + 债市崩盘)")
    if st.get('Curve') == 'red' and st.get('AI') == 'red':
        veto_msgs.append("衰退交易 (曲线深倒挂 + 科技崩盘)")

    if veto_msgs:
        overall_status = "🔴 系统性风险 (SYSTEM RISK)"
        summary_text = f"⚠️ 严重警报: {' + '.join(veto_msgs)}。建议清仓防御。"
        bg_color = "#fff5f5"
        header_color = "#c0392b"
    elif st.get('Credit') == 'red' or st.get('Curve') == 'red':
        overall_status = "🟠 结构性风险 (Structural Stress)"
        summary_text = "股市还没跌，但债市（聪明的钱）已经在跑路了，请高度警惕。"
        header_color = "#d35400"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>US Market Monitor</title>
        <style>
            body {{ font-family: 'Segoe UI', Roboto, sans-serif; background-color: {bg_color}; margin: 0; padding: 20px; color: #333; }}
            .container {{ max-width: 960px; margin: auto; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden; }}
            
            .header {{ background: {header_color}; color: white; padding: 25px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 22px; }}
            .timestamp {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}
            
            .status-box {{ padding: 20px; text-align: center; border-bottom: 1px solid #eee; }}
            .status-title {{ font-size: 24px; font-weight: bold; color: {header_color}; }}
            
            .grid {{ padding: 20px; display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); }}
            
            .card {{ border: 1px solid #eee; border-radius: 8px; overflow: hidden; background: #fff; }}
            .card-header {{ background: #f8f9fa; padding: 10px 15px; font-weight: bold; color: #555; font-size: 14px; border-bottom: 1px solid #eee; }}
            .card-body {{ padding: 15px; }}
            
            .item {{ display: flex; justify-content: space-between; margin-bottom: 20px; border-bottom: 1px dashed #f0f0f0; padding-bottom: 15px; }}
            .item:last-child {{ border: none; margin-bottom: 0; padding-bottom: 0; }}
            
            .info {{ flex: 1; margin-right: 15px; }}
            .item-name {{ font-weight: bold; font-size: 15px; color: #2c3e50; }}
            .item-desc {{ font-size: 13px; color: #777; margin: 4px 0; }}
            .item-rule {{ font-size: 12px; color: #444; background: #fff8e1; padding: 6px; border-radius: 4px; display: block; margin-top: 6px; line-height: 1.5; border-left: 3px solid #f1c40f; }}
            
            .stats {{ text-align: right; min-width: 90px; }}
            .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-size: 12px; font-weight: bold; margin-bottom: 5px; }}
            .red {{ background: #e74c3c; }} .orange {{ background: #f39c12; }} .yellow {{ background: #f1c40f; color: #444; }} .green {{ background: #27ae60; }} .gray {{ background: #ccc; }}
            
            .z-score {{ font-family: monospace; font-size: 14px; font-weight: bold; color: #2c3e50; }}
            .bias {{ font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🇺🇸 ESPT 美国宏观风险仪表盘 (逻辑修正版)</h1>
                <div class="timestamp">更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="status-box">
                <div class="status-title">{overall_status}</div>
                <div>{summary_text}</div>
            </div>
            
            <div class="grid">
    """
    
    for category, items in indicators.items():
        html += f"<div class='card'><div class='card-header'>{category}</div><div class='card-body'>"
        for item in items:
            z_val = item.get('z', 0)
            bias_val = item.get('bias', 0) * 100
            
            html += f"""
            <div class="item">
                <div class="info">
                    <div class="item-name">{item['name']}</div>
                    <div class="item-desc">{item['desc']}</div>
                    <div class="item-rule">{item['risk_rule']}</div>
                </div>
                <div class="stats">
                    <div class="badge {item['level']}">{item['text']}</div>
                    <div>Z: <span class="z-score">{z_val:+.2f}</span></div>
                    <div class="bias">B: {bias_val:+.1f}%</div>
                </div>
            </div>
            """
        html += "</div></div>"
        
    html += """
            </div>
            <div style="text-align:center; padding:15px; color:#999; font-size:12px;">
                Z-Score > 0 代表价格高于年线 (趋势向上) | Z-Score < 0 代表价格低于年线 (趋势向下)
            </div>
        </div>
    </body>
    </html>
    """
    
    filename = "usa_espt_optimized.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 报告已生成: {os.path.abspath(filename)}")

if __name__ == "__main__":
    try:
        data = get_us_indicators_optimized()
        generate_html_report(data)
        if os.name == 'nt': os.system("start us_market_monitor_fixed.html")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
