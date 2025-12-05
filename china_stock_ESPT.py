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
        # 【优化1】统一数据长度要求为年线的85%，兼顾稳定性与响应速度
        self.min_data_points = int(self.window_long * 0.85)
        # 【优化2】微调阈值，减少假警报 (2.0 -> 2.2)
        self.z_thresholds = {"red": 2.2, "orange": 1.2, "green": -1.0}
    
    def align_time_series(self, series1, series2):
        """智能对齐：解决A股/港股/美股休市日不同的问题"""
        if series1.index.tz: series1.index = series1.index.tz_localize(None)
        if series2.index.tz: series2.index = series2.index.tz_localize(None)
        
        # 取并集索引并前向填充 (Forward Fill)，确保不漏掉任何一方的交易日
        all_dates = series1.index.union(series2.index).sort_values()
        s1 = series1.reindex(all_dates).ffill()
        s2 = series2.reindex(all_dates).ffill()
        
        # 去除开头因填充产生的空值
        valid_mask = ~(s1.isna() | s2.isna())
        return s1[valid_mask], s2[valid_mask]

    def calculate_robust_z_score(self, series, inverse=False):
        """
        核心算法：乖离率 Z-Score (Bias Z-Score)
        逻辑：不仅看当前偏离了多少，还要看这个偏离程度在历史上是否罕见。
        """
        if len(series) < self.min_data_points: return 0, 0.0

        # 1. 计算年线 (Trend)
        rolling_mean = series.rolling(window=self.window_long, min_periods=self.min_data_points).mean()
        
        # 2. 计算乖离率 (Bias = Price / MA - 1)
        valid_idx = rolling_mean.index[~rolling_mean.isna()]
        if len(valid_idx) == 0: return 0, 0.0
        
        series_valid = series.loc[valid_idx]
        mean_valid = rolling_mean.loc[valid_idx]
        bias_series = (series_valid / mean_valid) - 1
        
        # 3. 计算乖离率的历史分布 (Mean & Std)
        # 这是为了解决"异方差"问题：将波动率不同的资产统一量纲
        bias_mean = bias_series.rolling(window=self.window_long).mean()
        bias_std = bias_series.rolling(window=self.window_long).std()
        
        # 4. 提取当前状态
        last_idx = bias_series.index[-1]
        cur_bias = bias_series.loc[last_idx]
        cur_mean = bias_mean.loc[last_idx]
        cur_std = bias_std.loc[last_idx]
        
        # 5. Z-Score 标准化
        if pd.isna(cur_std) or cur_std == 0: z_score = 0
        else: z_score = (cur_bias - cur_mean) / cur_std
            
        # 6. Winsorizing (防止极端数据破坏图表显示)
        z_score = np.clip(z_score, -4.5, 4.5)
        
        # 7. 风险方向调整 (Inverse=True 代表"跌是风险")
        risk_z = -z_score if inverse else z_score
        return risk_z, cur_bias

    def fetch_data_safe(self, ticker, period="5y"):
        """带重试机制的数据获取"""
        for _ in range(3): # 增加一次重试
            try:
                # 强制关闭auto_adjust以获取原始收盘价，有时更稳定
                df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
                if not df.empty and len(df) > 10: 
                    return df['Close']
            except: 
                time.sleep(1)
        return pd.Series(dtype=float)

    def fetch_and_analyze(self, name, rationale, ticker=None, 
                         inverse=False, is_ratio=False, 
                         ratio_num=None, ratio_den=None, 
                         fallback_ticker=None, external_series=None):
        try:
            series = None
            display_ticker = ticker
            
            # --- 模式A: 外部序列 ---
            if external_series is not None:
                series = external_series
                display_ticker = "Composite"
            
            # --- 模式B: 比率分析 (Pair Trading Logic) ---
            elif is_ratio:
                s_num = self.fetch_data_safe(ratio_num)
                s_den = self.fetch_data_safe(ratio_den)
                if s_num.empty or s_den.empty: raise ValueError("比率数据缺失")
                s_num, s_den = self.align_time_series(s_num, s_den)
                if len(s_num) < self.min_data_points: raise ValueError("比率数据长度不足")
                series = s_num / s_den
                display_ticker = f"{ratio_num}/{ratio_den}"
                
            # --- 模式C: 单资产模式 (带备用) ---
            else:
                series = self.fetch_data_safe(ticker)
                if (series.empty or len(series) < self.min_data_points) and fallback_ticker:
                    series = self.fetch_data_safe(fallback_ticker)
                    display_ticker = fallback_ticker
                if series.empty: raise ValueError("数据源失效")

            if series.index.tz: series.index = series.index.tz_localize(None)
            
            # 计算
            current_val = series.iloc[-1]
            z_score, bias = self.calculate_robust_z_score(series, inverse)
            
            # 评级逻辑
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
            return {"name": name, "value": 0, "bias": 0, "z": 0, "level": "gray", "text": "Error", "rationale": str(e)[:20]}

analyzer = MacroAnalyzer()

# ==========================================
# 2. 中国指标配置 (Optimized Sensors)
# ==========================================

def get_china_indicators():
    print("🔍 正在扫描中国股市 (China Offshore Proxies)...")
    indicators = {"E (预期)": [], "S (结构)": [], "P (权力)": [], "T (技术)": []}

    # --- E: 预期 (Sentiment / Expectation) ---
    # 【优化】替换PDD/BABA，使用互联网/蓝筹比率代表风险偏好
    # KWEB(科技成长) vs FXI(银行能源)。比率上升代表资金进攻，下跌代表防御。
    indicators["E (预期)"].append(analyzer.fetch_and_analyze(
        name="风险偏好 (KWEB/FXI)", is_ratio=True, ratio_num="KWEB", ratio_den="FXI",
        rationale="成长/价值比。比率暴跌=市场极度防御(悲观)。", inverse=True # 跌是风险
    ))

    # 大盘情绪 (FXI) - 最直接的离岸中国资产流动性指标
    indicators["E (预期)"].append(analyzer.fetch_and_analyze(
        name="大盘情绪 (FXI)", ticker="FXI", fallback_ticker="MCHI",
        rationale="富时中国A50 ETF。价格负乖离过大=恐慌抛售。", inverse=True # 跌是风险
    ))

    # --- S: 结构 (Structure) ---
    # 【优化】保留CHIR但增加CHIQ(可选消费)，构成"房产+消费"双结构
    indicators["S (结构)"].append(analyzer.fetch_and_analyze(
        name="地产板块 (CHIR)", ticker="CHIR", 
        rationale="房地产ETF。硬着陆风险的最真实反映(尽管流动性一般)。", inverse=True
    ))

    indicators["S (结构)"].append(analyzer.fetch_and_analyze(
        name="内需消费 (CHIQ)", ticker="CHIQ",
        rationale="可选消费ETF。持续走弱=内需不足的结构性确认。", inverse=True
    ))

    # --- P: 权力/宏观 (Power / Policy) ---
    # 汇率 - 央行的底线
    indicators["P (权力)"].append(analyzer.fetch_and_analyze(
        name="汇率压力 (USDCNY)", ticker="USDCNY=X", fallback_ticker="CNH=X",
        rationale="汇率急贬(正乖离)=资本外流压力，可能引发政策干预。", inverse=False # 涨是风险
    ))

    # 铜 - 实体经济/基建的真实需求
    indicators["P (权力)"].append(analyzer.fetch_and_analyze(
        name="工业需求 (铜)", ticker="HG=F", fallback_ticker="COPX",
        rationale="铜博士。价格暴跌=实体经济/基建失速风险。", inverse=True # 跌是风险
    ))

    # --- T: 技术 (Technology) ---
    # 科技竞争力 - 相对美股的强弱
    indicators["T (技术)"].append(analyzer.fetch_and_analyze(
        name="科技相对强弱 (CN/US)", is_ratio=True, ratio_num="CQQQ", ratio_den="SPY",
        rationale="CN科技跑输美股大盘=缺乏独立上涨逻辑。", inverse=True
    ))

    # 新能源 - 战略新兴产业
    indicators["T (技术)"].append(analyzer.fetch_and_analyze(
        name="新能源 (KGRN)", ticker="KGRN",
        rationale="新三样出口景气度。股价反映全球贸易环境。", inverse=True
    ))

    return indicators

# ==========================================
# 3. 报告生成 (Report & Fusion Logic)
# ==========================================

def generate_html_report(indicators):
    # 1. 熔断与状态计算 (Fusion Logic)
    st = {}
    for cat in indicators.values():
        for item in cat:
            if "地产" in item['name']: st['RealEstate'] = item['level']
            if "汇率" in item['name']: st['FX'] = item['level']
            if "大盘" in item['name']: st['Market'] = item['level']
            if "风险偏好" in item['name']: st['RiskOn'] = item['level']

    # 默认状态
    overall_status = "🟢 市场情绪平稳 (Stable)"
    summary_text = "宏观代理指标处于正常波动区间，未见显著系统性风险信号。"
    header_bg = "#c0392b" # 中国红
    body_bg = "#fdf2e9"   # 米色背景
    
    # --- 优化的熔断逻辑 ---
    veto_msgs = []
    
    # 逻辑1: "股汇双杀" (最典型的危机模式)
    if st.get('FX') == 'red' and st.get('Market') == 'red':
        veto_msgs.append("股汇双杀(资本外流+股市崩盘)")
        
    # 逻辑2: "资产负债表衰退" (地产崩盘 + 风险偏好极低)
    if st.get('RealEstate') == 'red' and st.get('RiskOn') == 'red':
        veto_msgs.append("资产负债表衰退(地产+科技共振下跌)")
        
    if veto_msgs:
        overall_status = "🔴 系统性熔断 (SYSTEM FAILURE)"
        summary_text = f"⚠️ 触发危机模式: {' + '.join(veto_msgs)}。建议清仓避险。"
        header_bg = "#641e16" # 深血红
        body_bg = "#fadbd8"
    
    # 逻辑3: 结构性高压 (没有全面崩盘，但核心指标报警)
    elif st.get('FX') == 'red' or st.get('RealEstate') == 'red':
        overall_status = "🟠 结构性警报 (Structural Stress)"
        risk_source = "汇率" if st.get('FX') == 'red' else "地产"
        summary_text = f"核心宏观锚点 ({risk_source}) 出现极度异常，市场极度脆弱。"
        header_bg = "#d35400" # 南瓜橙
        
    # 逻辑4: 超跌反弹机会 (大盘极度恐慌，但汇率稳定)
    elif st.get('Market') == 'red' and st.get('FX') in ['green', 'yellow']:
        overall_status = "🟢 黄金坑/超跌 (Oversold Opportunity)"
        summary_text = "股市出现恐慌性抛售，但汇率/宏观面稳定，可能存在反弹机会。"
        header_bg = "#229954" # 翡翠绿

    # 2. 生成HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>China Stock ESPT Dashboard (Optimized)</title>
        <style>
            body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; background-color: {body_bg}; padding: 20px; color: #333; }}
            .container {{ max-width: 960px; margin: auto; background: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden; }}
            .header {{ background: {header_bg}; color: #f4d03f; padding: 25px; text-align: center; border-bottom: 4px solid rgba(0,0,0,0.1); }}
            .header h1 {{ margin: 0; font-size: 24px; font-weight: 800; letter-spacing: 1px; }}
            .timestamp {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}
            
            .status-box {{ padding: 20px; text-align: center; border-bottom: 1px solid #eee; background: #fff; }}
            .status-title {{ font-size: 22px; font-weight: bold; color: {header_bg}; margin-bottom: 8px; }}
            .status-desc {{ color: #555; font-size: 14px; }}
            
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }}
            @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            
            .card {{ background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.03); transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.08); }}
            .card h3 {{ margin-top: 0; color: #c0392b; border-bottom: 2px solid #f2d7d5; padding-bottom: 8px; font-size: 15px; text-transform: uppercase; }}
            
            .item {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px dashed #f0f0f0; }}
            .item:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
            
            .label {{ font-weight: 700; font-size: 13px; color: #2c3e50; }}
            .rationale {{ font-size: 10px; color: #95a5a6; margin-top: 3px; max-width: 220px; }}
            
            .values {{ text-align: right; }}
            .main-val {{ font-weight: 700; font-size: 15px; font-family: Consolas, monospace; }}
            .sub-val {{ font-size: 10px; color: #7f8c8d; margin-top: 2px; }}
            
            .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; color: white; font-size: 10px; margin-left: 5px; vertical-align: middle; font-weight: bold; }}
            .red {{ background: #e74c3c; }} .orange {{ background: #e67e22; }} 
            .yellow {{ background: #f1c40f; color: #444; }} .green {{ background: #27ae60; }} .gray {{ background: #bdc3c7; }}
            
            .footer {{ padding: 15px; text-align: center; background: #fafafa; font-size: 11px; color: #aaa; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🇨🇳 ESPT 中国市场风险仪表盘 (Optimized)</h1>
                <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="status-box">
                <div class="status-title">{overall_status}</div>
                <div class="status-desc">{summary_text}</div>
            </div>
            <div class="grid">
    """
    
    for dim, items in indicators.items():
        html += f"<div class='card'><h3>{dim}</h3>"
        for item in items:
            bias_val = item.get('bias', 0) * 100
            bias_str = f"{bias_val:+.1f}%" if item.get('ticker') != "Error" else "-"
            val_str = f"{item.get('value', 0):.2f}"
            
            html += f"""
            <div class="item">
                <div>
                    <div class="label">{item['name']} <span class="badge {item['level']}">{item['text']}</span></div>
                    <div class="rationale">{item['rationale']}</div>
                </div>
                <div class="values">
                    <div class="main-val">{val_str}</div>
                    <div class="sub-val">Z: {item.get('z', 0):+.2f} | 乖离: {bias_str}</div>
                </div>
            </div>
            """
        html += "</div>"
        
    html += """
            </div>
            <div class="footer">
                <b>免责声明:</b> 本报告基于离岸ETF及衍生品数据 (Yahoo Finance) 生成，仅供参考。<br>
                核心算法: Robust Bias Z-Score Model (Win:252/0.85)
            </div>
        </div>
    </body>
    </html>
    """
    
    filename = "china_espt_optimized.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 报告已生成: {os.path.abspath(filename)}")

if __name__ == "__main__":
    try:
        data = get_china_indicators()
        generate_html_report(data)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

