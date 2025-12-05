import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# ==========================================
# 1. 核心计算引擎 (Optimized Bias Z-Score Engine)
# ==========================================

class MacroAnalyzer:
    def __init__(self):
        self.window_long = 252  # 1年交易日基准
        # 【优化1】统一数据长度要求为年线的85%
        self.min_data_points = int(self.window_long * 0.85)
        # 【优化2】阈值微调，适应日元资产的高波动性 (2.0 -> 2.2)
        self.z_thresholds = {"red": 2.2, "orange": 1.2, "green": -1.0}
    
    def align_time_series(self, series1, series2):
        """智能对齐：处理日股/美股休市日不同的问题"""
        if series1.index.tz: series1.index = series1.index.tz_localize(None)
        if series2.index.tz: series2.index = series2.index.tz_localize(None)
        
        all_dates = series1.index.union(series2.index).sort_values()
        s1 = series1.reindex(all_dates).ffill()
        s2 = series2.reindex(all_dates).ffill()
        
        valid_mask = ~(s1.isna() | s2.isna())
        return s1[valid_mask], s2[valid_mask]

    def calculate_robust_z_score(self, series, inverse=False):
        """核心算法：乖离率 Z-Score"""
        if len(series) < self.min_data_points: return 0, 0.0

        # 1. 计算年线
        rolling_mean = series.rolling(window=self.window_long, min_periods=self.min_data_points).mean()
        
        # 2. 计算乖离率 (Bias)
        valid_idx = rolling_mean.index[~rolling_mean.isna()]
        if len(valid_idx) == 0: return 0, 0.0
        
        series_valid = series.loc[valid_idx]
        mean_valid = rolling_mean.loc[valid_idx]
        bias_series = (series_valid / mean_valid) - 1
        
        # 3. Z-Score 标准化
        bias_mean = bias_series.rolling(window=self.window_long).mean()
        bias_std = bias_series.rolling(window=self.window_long).std()
        
        last_idx = bias_series.index[-1]
        cur_bias = bias_series.loc[last_idx]
        cur_mean = bias_mean.loc[last_idx]
        cur_std = bias_std.loc[last_idx]
        
        if pd.isna(cur_std) or cur_std == 0: z_score = 0
        else: z_score = (cur_bias - cur_mean) / cur_std
            
        # Winsorizing
        z_score = np.clip(z_score, -4.5, 4.5)
        
        # 风险方向 (Inverse=True: 跌是风险)
        risk_z = -z_score if inverse else z_score
        return risk_z, cur_bias

    def fetch_data_safe(self, ticker, period="2y"):
        """带重试的数据获取"""
        for _ in range(3):
            try:
                df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
                if not df.empty and len(df) > 10: return df['Close']
            except: time.sleep(1)
        return pd.Series(dtype=float)

    def fetch_and_analyze(self, name, rationale, ticker=None, 
                         inverse=False, external_series=None, fallback_ticker=None):
        try:
            series = None
            display_ticker = ticker
            
            if external_series is not None:
                series = external_series
                display_ticker = "Composite"
            else:
                series = self.fetch_data_safe(ticker)
                if (series.empty or len(series) < self.min_data_points) and fallback_ticker:
                    series = self.fetch_data_safe(fallback_ticker)
                    display_ticker = fallback_ticker
                if series.empty: raise ValueError("Data Error")

            if series.index.tz: series.index = series.index.tz_localize(None)
            current_val = series.iloc[-1]
            z_score, bias = self.calculate_robust_z_score(series, inverse)
            
            if z_score > self.z_thresholds["red"]: level, text = "red", "极度异常"
            elif z_score > self.z_thresholds["orange"]: level, text = "orange", "显著偏离"
            elif z_score < self.z_thresholds["green"]: level, text = "green", "低位安全"
            else: level, text = "yellow", "均值回归"
            
            return {
                "name": name, "value": current_val, "bias": bias,
                "z": z_score, "level": level, "text": text, 
                "rationale": rationale, "ticker": display_ticker
            }
        except Exception as e:
            return {"name": name, "value": 0, "level": "gray", "text": "Error", "rationale": str(e)[:20]}

analyzer = MacroAnalyzer()

# ==========================================
# 2. 日本指标配置 (Optimized Sensors)
# ==========================================

def get_japan_indicators():
    print("🔍 正在扫描日本股市 (Japan Real-Time Data)...")
    indicators = {"E (预期)": [], "S (结构)": [], "P (权力)": [], "T (技术)": []}

    # --- E: 预期 (Sentiment) ---
    # 1. 恐慌指数 (N225 Volatility)
    try:
        n225 = analyzer.fetch_data_safe("^N225")
        if not n225.empty:
            # 手动计算20日滚动波动率
            returns = np.log(n225 / n225.shift(1))
            vol = returns.rolling(20).std() * np.sqrt(252) * 100
            indicators["E (预期)"].append(analyzer.fetch_and_analyze(
                name="恐慌指数 (N225 Vol)", external_series=vol,
                rationale="日经波动率。飙升(正乖离)=市场恐慌。", inverse=False
            ))
        else: raise ValueError
    except:
        indicators["E (预期)"].append({"name": "恐慌指数", "value": 0, "level": "gray", "text": "Error"})

    # 2. 输入性通胀 (Pain Index = Oil * Yen) - 核心原创指标，保留
    try:
        oil = analyzer.fetch_data_safe("CL=F")
        yen = analyzer.fetch_data_safe("USDJPY=X")
        if not oil.empty and not yen.empty:
            oil, yen = analyzer.align_time_series(oil, yen)
            pain = oil * yen
            indicators["E (预期)"].append(analyzer.fetch_and_analyze(
                name="家庭痛苦指数 (Oil*Yen)", external_series=pain,
                rationale="油价与汇率双升=购买力缩水，利空消费。", inverse=False # 涨是痛苦(风险)
            ))
        else: raise ValueError
    except:
        indicators["E (预期)"].append({"name": "痛苦指数", "value": 0, "level": "gray", "text": "Error"})

    # --- S: 结构 (Structure) ---
    # 1. 【优化】替换优衣库，使用东证REITs指数
    indicators["S (结构)"].append(analyzer.fetch_and_analyze(
        name="通胀预期 (东证REITs)", ticker="1343.T", fallback_ticker="TREIT",
        rationale="房地产信托ETF。上涨确认国内资产通胀逻辑，下跌则为通缩回归。", inverse=True # 跌是风险(通缩)
    ))

    # 2. 央行博弈 (三菱日联 8306.T) - 保留
    indicators["S (结构)"].append(analyzer.fetch_and_analyze(
        name="加息押注 (三菱日联)", ticker="8306.T", fallback_ticker="MUFG",
        rationale="银行股暴涨(正乖离)=市场押注YCC取消/加息，利空债市。", inverse=False # 暴涨是系统性风险
    ))

    # --- P: 权力 (Power / BOJ) ---
    # 1. 汇率干预线 (USDJPY)
    indicators["P (权力)"].append(analyzer.fetch_and_analyze(
        name="汇率风险 (USDJPY)", ticker="USDJPY=X",
        rationale="日元急贬(正乖离)=央行干预风险剧增。", inverse=False # 涨是风险
    ))

    # 2. 外资风向 (三菱商事 8058.T) - 巴菲特指标
    indicators["P (权力)"].append(analyzer.fetch_and_analyze(
        name="外资风向 (三菱商事)", ticker="8058.T", fallback_ticker="8031.T", # 备用三井物产
        rationale="五大商社是外资配置日股的风向标。下跌=外资撤退。", inverse=True # 跌是风险
    ))

    # --- T: 技术 (Technology) ---
    # 1. 半导体周期 (东京电子 8035.T)
    indicators["T (技术)"].append(analyzer.fetch_and_analyze(
        name="AI/半导体 (东京电子)", ticker="8035.T",
        rationale="日本半导体设备龙头。下跌=全球AI周期见顶。", inverse=True
    ))

    # 2. 全球资本开支 (Fanuc 6954.T)
    indicators["T (技术)"].append(analyzer.fetch_and_analyze(
        name="工业机器人 (Fanuc)", ticker="6954.T",
        rationale="全球制造业Capex(资本开支)的最敏感指标。", inverse=True
    ))

    return indicators

# ==========================================
# 3. 报告生成 (Fusion Logic)
# ==========================================

def generate_html_report(indicators):
    # 1. 熔断逻辑
    st = {}
    for cat in indicators.values():
        for item in cat:
            if "痛苦" in item['name']: st['Pain'] = item['level']
            if "加息" in item['name']: st['Bank'] = item['level'] # 三菱日联
            if "汇率" in item['name']: st['Yen'] = item['level']
            if "REITs" in item['name']: st['Reits'] = item['level']

    # 默认状态
    overall_status = "🟢 市场环境良好 (Positive)"
    summary_text = "宏观指标平稳。通胀温和，汇率处于可控区间，外资情绪稳定。"
    header_bg = "#bc002d" # 日本红
    body_bg = "#f9f9f9"
    
    # --- 优化的熔断逻辑 ---
    veto_msgs = []
    
    # 逻辑1: 汇率失控 (Yen collapse)
    if st.get('Yen') == 'red':
        veto_msgs.append("汇率失控(干预风险)")
        
    # 逻辑2: 滞胀+加息双杀 (Stagflation Shock)
    # 痛苦指数飙升(通胀) + 银行股暴涨(加息预期) = 实体经济崩溃
    if st.get('Pain') == 'red' and st.get('Bank') == 'red':
        veto_msgs.append("滞胀+加息双杀")
        
    # 逻辑3: 通缩回归 (Deflation Return)
    # REITs崩盘 = 资产通胀故事破灭
    if st.get('Reits') == 'red':
        veto_msgs.append("通缩回归(REITs崩盘)")

    if veto_msgs:
        overall_status = "🔴 系统性熔断 (SYSTEM FAILURE)"
        summary_text = f"⚠️ 触发机制: {' + '.join(veto_msgs)}。建议回避日股，持有现金。"
        body_bg = "#fff0f0"
    
    # 逻辑4: 结构性高压 (High Stress)
    # 痛苦指数红了，或者银行股红了，但还没共振
    elif st.get('Pain') == 'red' or st.get('Bank') == 'red':
        overall_status = "🟠 结构性高压 (High Stress)"
        summary_text = "部分宏观因子(通胀/利率)出现极端乖离，市场波动率将显著上升。"
        header_bg = "#e67e22" # 橙色

    # 2. 生成HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Japan Stock ESPT Dashboard (Optimized)</title>
        <style>
            body {{ font-family: "Hiragino Kaku Gothic Pro", "Meiryo", sans-serif; background-color: {body_bg}; padding: 20px; color: #333; }}
            .container {{ max-width: 960px; margin: auto; background: white; border: 1px solid #ddd; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-radius: 4px; }}
            .header {{ background: {header_bg}; color: white; padding: 30px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 26px; letter-spacing: 2px; }}
            .timestamp {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}
            
            .status-box {{ padding: 25px; text-align: center; border-bottom: 1px solid #eee; background: #fff; }}
            .status-title {{ font-size: 22px; font-weight: bold; color: {header_bg}; margin-bottom: 10px; }}
            
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }}
            @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            
            .card {{ padding: 15px; border: 1px solid #eee; background: #fff; }}
            .card h3 {{ margin-top: 0; color: #333; font-size: 15px; border-left: 4px solid {header_bg}; padding-left: 10px; }}
            
            .item {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px dotted #eee; }}
            .item:last-child {{ border-bottom: none; }}
            
            .label {{ font-weight: 600; font-size: 14px; }}
            .rationale {{ font-size: 10px; color: #888; margin-top: 3px; }}
            
            .values {{ text-align: right; }}
            .main-val {{ font-weight: bold; font-size: 16px; font-family: monospace; }}
            .sub-val {{ font-size: 11px; color: #666; }}
            
            .tag {{ padding: 2px 6px; border-radius: 2px; font-size: 10px; color: white; margin-left: 5px; }}
            .red {{ background: #c0392b; }} .orange {{ background: #e67e22; }} 
            .yellow {{ background: #f1c40f; color: #333; }} .green {{ background: #27ae60; }} .gray {{ background: #95a5a6; }}
            
            .footer {{ padding: 15px; text-align: center; background: #f4f4f4; font-size: 11px; color: #777; border-top: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🇯🇵 ESPT 日本股票风险仪表盘 (Optimized)</h1>
                <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="status-box">
                <div class="status-title">{overall_status}</div>
                <div>{summary_text}</div>
            </div>
            <div class="grid">
    """
    
    for dim, items in indicators.items():
        html += f"<div class='card'><h3>{dim}</h3>"
        for item in items:
            bias_val = item.get('bias', 0) * 100
            bias_str = f"{bias_val:+.1f}%" if item.get('ticker') != "Error" else "-"
            
            html += f"""
            <div class="item">
                <div>
                    <div class="label">{item['name']} <span class="tag {item['level']}">{item['text']}</span></div>
                    <div class="rationale">{item['rationale']}</div>
                </div>
                <div class="values">
                    <div class="main-val">{item.get('value', 0):.2f}</div>
                    <div class="sub-val">Z: {item.get('z', 0):+.2f} | 乖离: {bias_str}</div>
                </div>
            </div>
            """
        html += "</div>"
        
    html += """
            </div>
            <div class="footer">
                数据源: Yahoo Finance | 算法: Bias Z-Score (Win:252/0.85)
            </div>
        </div>
    </body>
    </html>
    """
    
    filename = "japan_espt_optimized.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 报告已生成: {os.path.abspath(filename)}")

if __name__ == "__main__":
    try:
        data = get_japan_indicators()
        generate_html_report(data)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
