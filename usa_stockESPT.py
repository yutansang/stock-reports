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
        # 【优化1】统一参数：年线的85%。
        # 美国版原先的1.2倍(300天)过于苛刻，容易导致新指标无法计算。
        self.min_data_points = int(self.window_long * 0.85)
        
        # 【优化2】阈值微调：从2.0上调至2.2。
        # 美股特别是VIX波动剧烈，提高阈值能减少"狼来了"的误报。
        self.z_thresholds = {"red": 2.2, "orange": 1.2, "green": -1.0}
    
    def align_time_series(self, series1, series2):
        """智能对齐：处理两个序列日期不一致的问题"""
        if series1.index.tz: series1.index = series1.index.tz_localize(None)
        if series2.index.tz: series2.index = series2.index.tz_localize(None)
        
        all_dates = series1.index.union(series2.index).sort_values()
        s1 = series1.reindex(all_dates).ffill()
        s2 = series2.reindex(all_dates).ffill()
        
        valid_mask = ~(s1.isna() | s2.isna())
        return s1[valid_mask], s2[valid_mask]

    def calculate_robust_z_score(self, series, inverse=False):
        """
        核心算法：乖离率 Z-Score
        """
        if len(series) < self.min_data_points: return 0, 0.0

        # 1. 计算年线
        rolling_mean = series.rolling(window=self.window_long, min_periods=self.min_data_points).mean()
        
        # 2. 计算乖离率 (Bias)
        valid_idx = rolling_mean.index[~rolling_mean.isna()]
        if len(valid_idx) == 0: return 0, 0.0
        
        series_valid = series.loc[valid_idx]
        mean_valid = rolling_mean.loc[valid_idx]
        bias_series = (series_valid / mean_valid) - 1
        
        # 3. Z-Score 标准化 (解决异方差问题)
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
        
        # 风险方向 (Inverse=True: 跌是风险; Inverse=False: 涨是风险)
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
                         inverse=False, is_ratio=False, 
                         ratio_num=None, ratio_den=None, 
                         fallback_ticker=None, external_series=None):
        try:
            series = None
            display_ticker = ticker
            
            # 模式A: 外部序列
            if external_series is not None:
                series = external_series
                display_ticker = "Composite"
            # 模式B: 比率分析
            elif is_ratio:
                s_num = self.fetch_data_safe(ratio_num)
                s_den = self.fetch_data_safe(ratio_den)
                if s_num.empty or s_den.empty: raise ValueError("比率数据缺失")
                s_num, s_den = self.align_time_series(s_num, s_den)
                if len(s_num) < self.min_data_points: raise ValueError("长度不足")
                series = s_num / s_den
                display_ticker = f"{ratio_num}/{ratio_den}"
            # 模式C: 单资产
            else:
                series = self.fetch_data_safe(ticker)
                if (series.empty or len(series) < self.min_data_points) and fallback_ticker:
                    series = self.fetch_data_safe(fallback_ticker)
                    display_ticker = fallback_ticker
                if series.empty: raise ValueError("数据失效")

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
# 2. 美国指标配置 (Optimized Sensors)
# ==========================================

def get_us_indicators():
    print("🔍 正在扫描美国股市 (US Real-Time Data)...")
    indicators = {"E (预期)": [], "S (结构)": [], "P (权力)": [], "T (技术)": []}

    # --- E: 预期 (Sentiment) ---
    # 1. 恐慌指数 (VIX)
    # 修正：使用 VIXY 作为备用，以防 ^VIX 数据延迟
    indicators["E (预期)"].append(analyzer.fetch_and_analyze(
        name="恐慌指数 (VIX)", ticker="^VIX", fallback_ticker="VIXY",
        rationale="华尔街恐惧指标。正乖离过大(飙升)=市场极度恐慌。", inverse=False # 涨是风险
    ))
    
    # 2. 贪婪/防御 (XLY/XLP)
    indicators["E (预期)"].append(analyzer.fetch_and_analyze(
        name="贪婪/防御 (XLY/XLP)", is_ratio=True, ratio_num="XLY", ratio_den="XLP",
        rationale="可选/必选消费比。比率下行=资金涌入防御板块避险。", inverse=True # 跌是风险
    ))

    # --- S: 结构 (Structure) ---
    # 1. 市场广度 (RSP/SPY) - 确认保留
    indicators["S (结构)"].append(analyzer.fetch_and_analyze(
        name="市场广度 (RSP/SPY)", is_ratio=True, ratio_num="RSP", ratio_den="SPY",
        rationale="等权/市值比。比率下行=涨势只集中在巨头，市场脆弱。", inverse=True # 跌是风险
    ))

    # 2. 信用风险 (HYG) - 确认保留
    indicators["S (结构)"].append(analyzer.fetch_and_analyze(
        name="信用底座 (HYG)", ticker="HYG", fallback_ticker="JNK",
        rationale="垃圾债ETF。价格崩盘(负乖离)=企业融资环境恶化。", inverse=True # 跌是风险
    ))

    # --- P: 权力 (Power/Fed) ---
    # 1. 收益率曲线 (10Y-2Y)
    try:
        t10 = analyzer.fetch_data_safe("^TNX")
        t2 = analyzer.fetch_data_safe("^FVX")
        if not t10.empty and not t2.empty:
            t10, t2 = analyzer.align_time_series(t10, t2)
            spread = t10 - t2
            # 注意：倒挂加深是风险。Spread越小(越负)越危险。
            # 这里的Z-Score逻辑：如果Spread异常低(负乖离)，Z为负，Risk_Z变正(Red)。
            indicators["P (权力)"].append(analyzer.fetch_and_analyze(
                name="收益率曲线 (10Y-2Y)", external_series=spread,
                rationale="衰退预警最准指标。倒挂加深(负值变大)=衰退逼近。", inverse=True # 低是风险
            ))
        else: raise ValueError
    except:
        indicators["P (权力)"].append({"name": "收益率曲线", "value": 0, "level": "gray", "text": "Error"})

    # 2. 流动性代理 (BTC) - 确认保留 (工程最优解)
    indicators["P (权力)"].append(analyzer.fetch_and_analyze(
        name="边际流动性 (BTC)", ticker="BTC-USD",
        rationale="对美元流动性最敏感的7x24资产。暴跌=流动性收紧。", inverse=True # 跌是风险
    ))

    # --- T: 技术 (Technology) ---
    # 1. 科技拥挤度 (QQQ/SPY)
    indicators["T (技术)"].append(analyzer.fetch_and_analyze(
        name="科技拥挤度 (QQQ/SPY)", is_ratio=True, ratio_num="QQQ", ratio_den="SPY",
        rationale="比率过高(正乖离)=交易过度拥挤，随时可能踩踏。", inverse=False # 高是风险
    ))

    # 2. AI硬件引擎 (SMH)
    indicators["T (技术)"].append(analyzer.fetch_and_analyze(
        name="AI引擎 (SMH)", ticker="SMH",
        rationale="半导体周期。跌破年线(负乖离)=牛市发动机熄火。", inverse=True # 跌是风险
    ))

    return indicators

# ==========================================
# 3. 报告生成 (Fusion Logic)
# ==========================================

def generate_html_report(indicators):
    # 1. 计算熔断逻辑
    st = {}
    for cat in indicators.values():
        for item in cat:
            if "VIX" in item['name']: st['VIX'] = item['level']
            if "HYG" in item['name']: st['Credit'] = item['level']
            if "收益率" in item['name']: st['Curve'] = item['level']
            if "SMH" in item['name']: st['AI'] = item['level']
    
    # 默认设置
    overall_status = "🟢 趋势健康 (Healthy Trend)"
    summary_text = "各项宏观指标处于正常波动区间 (Goldilocks)，适合顺势而为。"
    bg_color = "#f4f6f7" # 淡蓝灰
    header_color = "#2c3e50" # 深蓝
    
    # --- 优化的熔断逻辑 ---
    veto_msgs = []

    # 场景1: 流动性危机 (2008/2020模式)
    # VIX飙升 + 信用债(HYG)崩盘。这是最危险的信号。
    if st.get('VIX') == 'red' and st.get('Credit') == 'red':
        veto_msgs.append("流动性危机 (VIX+Credit共振)")
    
    # 场景2: 衰退实质化 (Recession Realized)
    # 收益率曲线异常 + 周期股(SMH)崩盘
    if st.get('Curve') == 'red' and st.get('AI') == 'red':
        veto_msgs.append("衰退交易 (曲线+科技崩盘)")

    if veto_msgs:
        overall_status = "🔴 系统性风险 (SYSTEM RISK)"
        summary_text = f"⚠️ 触发熔断机制: {' + '.join(veto_msgs)}。建议清仓或全面防御。"
        bg_color = "#fdedec" # 淡红
        header_color = "#c0392b" # 深红
        
    # 场景3: 高压震荡 (High Stress)
    # VIX还没红，但信用债或曲线已经红了
    elif st.get('Credit') == 'red' or st.get('Curve') == 'red':
        overall_status = "🟠 结构性预警 (Structural Stress)"
        summary_text = "虽然恐慌指数(VIX)尚未失控，但债券市场(信用/利率)已发出强烈警报。"
        header_color = "#d35400" # 橙色
        
    # 场景4: 黄金坑 (Oversold)
    # 科技股杀跌(AI Red)，但信用(Credit Green)和VIX(Green/Yellow)正常
    # 说明是杀估值，不是杀逻辑。
    elif st.get('AI') == 'red' and st.get('Credit') in ['green', 'yellow']:
        overall_status = "🟢 超跌机会 (Oversold Opportunity)"
        summary_text = "科技股出现深度回调，但信贷市场情绪稳定，可能存在错杀机会。"
        header_color = "#27ae60" # 绿色

    # 2. 生成HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>US Stock ESPT Dashboard (Optimized)</title>
        <style>
            body {{ font-family: "Segoe UI", "Roboto", sans-serif; background-color: {bg_color}; padding: 20px; }}
            .container {{ max-width: 960px; margin: auto; background: white; border-radius: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); overflow: hidden; }}
            .header {{ background: {header_color}; color: white; padding: 30px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 26px; font-weight: 600; }}
            .timestamp {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}
            
            .status-box {{ padding: 25px; text-align: center; border-bottom: 1px solid #eee; }}
            .status-title {{ font-size: 24px; font-weight: bold; color: {header_color}; margin-bottom: 10px; }}
            
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; padding: 25px; }}
            @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            
            .card {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; }}
            .card h3 {{ margin-top: 0; color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 12px; font-size: 16px; letter-spacing: 1px; }}
            
            .item {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px dashed #eee; }}
            .item:last-child {{ border-bottom: none; }}
            
            .label-group {{ flex: 1; }}
            .label {{ font-weight: 600; font-size: 14px; color: #2c3e50; display: flex; align-items: center; }}
            .rationale {{ font-size: 11px; color: #7f8c8d; margin-top: 4px; max-width: 250px; line-height: 1.4; }}
            
            .values {{ text-align: right; }}
            .main-val {{ font-weight: bold; font-size: 16px; font-family: monospace; }}
            .sub-val {{ font-size: 11px; color: #95a5a6; margin-top: 2px; }}
            
            .dot {{ height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 8px; }}
            .red {{ color: #c0392b; }} .red .dot {{ background: #c0392b; }}
            .orange {{ color: #e67e22; }} .orange .dot {{ background: #e67e22; }}
            .yellow {{ color: #f1c40f; }} .yellow .dot {{ background: #f1c40f; }}
            .green {{ color: #27ae60; }} .green .dot {{ background: #27ae60; }}
            .gray {{ color: #95a5a6; }} .gray .dot {{ background: #95a5a6; }}
            
            .footer {{ padding: 20px; text-align: center; color: #999; font-size: 11px; background: #f8f9fa; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🇺🇸 ESPT 美国股票市场风险仪表盘 (Optimized)</h1>
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
                <div class="label-group">
                    <div class="label {item['level']}"><span class="dot"></span>{item['name']}</div>
                    <div class="rationale">{item['rationale']}</div>
                </div>
                <div class="values">
                    <div class="main-val {item['level']}">{item['text']}</div>
                    <div class="sub-val">Z: {item.get('z', 0):+.2f} | 乖离: {bias_str}</div>
                </div>
            </div>
            """
        html += "</div>"
        
    html += """
            </div>
            <div class="footer">
                Data Source: Yahoo Finance (Real-time) | Algorithm: Robust Bias Z-Score (Win:252/0.85, Threshold:2.2)
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
        data = get_us_indicators()
        generate_html_report(data)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

