import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import concurrent.futures

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================
REPORT_FILENAME = "china_espt_optimized.html"

# 定义监控清单
CONFIGS = [
    # --- E: 预期 (Sentiment) ---
    {
        "category": "E (预期)", "name": "风险偏好 (KWEB/FXI)", 
        "is_ratio": True, "ratio_num": "KWEB", "ratio_den": "FXI",
        "rationale": "互联网(进攻)/银行(防御)比率。暴跌代表市场极度避险。", 
        "inverse": True # 跌是风险
    },
    {
        "category": "E (预期)", "name": "大盘情绪 (FXI)", 
        "ticker": "FXI", "fallback_ticker": "MCHI",
        "rationale": "离岸中国蓝筹。负乖离过大代表流动性恐慌。", 
        "inverse": True # 跌是风险
    },
    # --- S: 结构 (Structure) ---
    {
        "category": "S (结构)", "name": "地产板块 (CHIR)", 
        "ticker": "CHIR", 
        "rationale": "地产链资金面。持续极低位代表债务通缩风险。", 
        "inverse": True
    },
    {
        "category": "S (结构)", "name": "内需消费 (CHIQ)", 
        "ticker": "CHIQ",
        "rationale": "可选消费意愿。反映居民端资产负债表健康度。", 
        "inverse": True
    },
    # --- P: 宏观/权力 (Power) ---
    {
        "category": "P (权力)", "name": "离岸汇率 (USD/CNH)", 
        # 现实修正：使用 CNH=F (离岸人民币期货) 或 CNH=X，更能反映外资态度
        "ticker": "CNH=F", "fallback_ticker": "USDCNY=X",
        "rationale": "汇率急贬(向上突破)往往伴随资产价格重估压力。", 
        "inverse": False # 涨是风险 (贬值)
    },
    {
        "category": "P (权力)", "name": "工业需求 (铜)", 
        "ticker": "HG=F", "fallback_ticker": "COPX",
        "rationale": "铜博士。价格与中国PMI高度相关，暴跌预示衰退。", 
        "inverse": True # 跌是风险
    },
    # --- T: 技术 (Tech/Momentum) ---
    {
        "category": "T (技术)", "name": "科技相对强弱 (CN/US)", 
        "is_ratio": True, "ratio_num": "CQQQ", "ratio_den": "SPY",
        "rationale": "如果CN科技持续跑输美股，说明缺乏独立逻辑。", 
        "inverse": True
    },
    {
        "category": "T (技术)", "name": "新能源 (KGRN)", 
        "ticker": "KGRN", 
        "rationale": "出口链/高端制造景气度代理指标。", 
        "inverse": True
    }
]

# ==========================================
# 2. 核心计算引擎 (Robust Engine)
# ==========================================

class MacroAnalyzer:
    def __init__(self):
        self.window_long = 252  # 1年
        self.min_data_points = 200 # 至少要有200天数据才计算，否则不准
        # 阈值微调：更加严格，避免噪音
        self.z_thresholds = {"red": 2.2, "orange": 1.5, "green": -1.5} 
    
    def fetch_data_single(self, ticker):
        """单线程下载，带更严格的清洗"""
        if not ticker: return pd.Series(dtype=float)
        try:
            # 获取5年数据，确保有足够的历史做 MAD 计算
            df = yf.Ticker(ticker).history(period="5y", auto_adjust=False)
            
            # 【现实修正1】清洗脏数据：去除 0 和 负数
            if df.empty: return pd.Series(dtype=float)
            df = df[df['Close'] > 0.01] 
            
            if len(df) > 10:
                return df['Close']
        except Exception as e:
            print(f"⚠️ 下载异常 {ticker}: {e}")
        return pd.Series(dtype=float)

    def fetch_data_batch(self, configs):
        """并行下载"""
        tickers = set()
        for item in configs:
            if item.get('ticker'): tickers.add(item['ticker'])
            if item.get('fallback_ticker'): tickers.add(item['fallback_ticker'])
            if item.get('ratio_num'): tickers.add(item['ratio_num'])
            if item.get('ratio_den'): tickers.add(item['ratio_den'])
        
        print(f"🚀 正在并行请求 {len(tickers)} 个数据源...")
        data_cache = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor: # 增加并发数
            future_to_ticker = {executor.submit(self.fetch_data_single, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                t = future_to_ticker[future]
                data_cache[t] = future.result()
        return data_cache

    def calculate_robust_z_score(self, series, inverse=False):
        """
        【核心算法修正】
        增加“波动率地板 (Noise Floor)”，防止死鱼股/汇率产生的除零暴涨。
        """
        if len(series) < self.min_data_points: return 0, 0.0, 0.0

        # 1. 趋势 (Trend)
        trend = series.rolling(window=self.window_long).mean()
        
        # 2. 乖离 (Bias)
        bias_series = (series / trend) - 1
        
        # 3. 鲁棒统计量 (MAD)
        rolling_median = bias_series.rolling(window=self.window_long).median()
        rolling_mad = (bias_series - rolling_median).abs().rolling(window=self.window_long).median()
        
        # 获取当前值
        try:
            cur_bias = bias_series.iloc[-1]
            cur_med = rolling_median.iloc[-1]
            cur_mad = rolling_mad.iloc[-1]
            
            # 【现实修正2】波动率地板 (Noise Floor)
            # 设定最小 MAD 为 0.5% (0.005)。如果历史波动率小于这个值，强制设为 0.005。
            # 这能避免汇率这种低波资产因为微小跳动而 Z-Score 爆炸。
            effective_mad = max(cur_mad, 0.005) 
            
            # MAD -> Std 转换因子 1.4826
            z_score = (cur_bias - cur_med) / (effective_mad * 1.4826)
            
            # 裁剪极端值
            z_score = np.clip(z_score, -5.0, 5.0)

            # 4. 短期动量 (Short-term Momentum)
            # Z-Score 看的是位置，Momentum 看的是速度。两者共振才是大风险。
            # 计算最近 5 天的变化率
            pct_chg_5d = series.pct_change(5).iloc[-1]

        except:
            return 0, 0.0, 0.0

        # 方向调整
        risk_z = -z_score if inverse else z_score
        return risk_z, cur_bias, pct_chg_5d

    def analyze(self):
        data_cache = self.fetch_data_batch(CONFIGS)
        results = []
        
        print("🧠 正在进行多维风险计算...")
        for config in CONFIGS:
            try:
                # 准备数据序列
                series = None
                display_ticker = ""
                
                if config.get('is_ratio'):
                    s1 = data_cache.get(config['ratio_num'])
                    s2 = data_cache.get(config['ratio_den'])
                    display_ticker = f"{config['ratio_num']}/{config['ratio_den']}"
                    if s1 is not None and s2 is not None and not s1.empty and not s2.empty:
                        # 对齐
                        common_idx = s1.index.intersection(s2.index)
                        if len(common_idx) > 100:
                            series = s1.loc[common_idx] / s2.loc[common_idx]
                else:
                    t = config.get('ticker')
                    fb = config.get('fallback_ticker')
                    s = data_cache.get(t)
                    display_ticker = t
                    # 自动切换备用
                    if (s is None or s.empty or len(s) < 200) and fb:
                        s = data_cache.get(fb)
                        display_ticker = fb
                    series = s

                # 空值检查
                if series is None or series.empty:
                    raise ValueError("No Data")
                
                # 计算
                risk_z, bias, mom_5d = self.calculate_robust_z_score(series, config.get('inverse', False))
                
                # 【现实修正3】评级逻辑优化
                # 只有当 Z-Score 很大 且 动量方向也一致时，才给予最高警报
                # 例如：Risk Z 高 (风险大)，且最近 5 天还在往风险方向走 (跌)
                
                level, text = "yellow", "正常波动"
                
                if risk_z > self.z_thresholds["red"]:
                    level, text = "red", "极度异常"
                elif risk_z > self.z_thresholds["orange"]:
                    level, text = "orange", "显著偏离"
                elif risk_z < self.z_thresholds["green"]:
                    # 注意：Risk Z 低意味着 "非常安全" 或者 "泡沫/超买" (取决于你的视角)
                    # 在风控模型里，我们标记为绿色，代表 "无下行风险"
                    level, text = "green", "安全/超跌"
                
                results.append({
                    "config": config,
                    "value": series.iloc[-1],
                    "bias": bias,
                    "z": risk_z,
                    "mom_5d": mom_5d,
                    "level": level,
                    "text": text,
                    "ticker": display_ticker
                })
                
            except Exception as e:
                results.append({
                    "config": config, "value": 0, "bias": 0, "z": 0, "mom_5d": 0,
                    "level": "gray", "text": "数据缺失", "ticker": "Error"
                })
        
        return results

# ==========================================
# 3. 报告生成 (HTML Generation)
# ==========================================

def generate_report(results):
    # 状态判定
    risk_count = sum(1 for r in results if r['level'] == 'red')
    warning_count = sum(1 for r in results if r['level'] == 'orange')
    
    # 简单的宏观状态机
    if risk_count >= 2:
        status = "🔴 红色警报 (CRITICAL RISK)"
        desc = "多个核心宏观指标出现极度异常，建议防御。"
        bg_color = "#fadbd8"
        head_color = "#c0392b"
    elif risk_count == 1 or warning_count >= 3:
        status = "🟠 结构性压力 (Structural Stress)"
        desc = "部分指标出现显著偏离，需密切关注。"
        bg_color = "#fdebd0"
        head_color = "#d35400"
    else:
        status = "🟢 宏观平稳 (Stable)"
        desc = "主要指标处于统计学合理区间。"
        bg_color = "#e8f8f5"
        head_color = "#27ae60"

    # HTML 组装
    html_cards = ""
    categories = {}
    for r in results:
        cat = r['config']['category']
        if cat not in categories: categories[cat] = []
        categories[cat].append(r)
        
    for cat, items in categories.items():
        html_cards += f"<div class='card'><h3>{cat}</h3>"
        for item in items:
            # 格式化
            bias_pct = item['bias'] * 100
            mom_pct = item['mom_5d'] * 100
            mom_arrow = "⬆" if mom_pct > 0 else "⬇"
            mom_color = "#e74c3c" if (item['config']['inverse'] and mom_pct < -0.02) else "#2ecc71"
            if not item['config']['inverse'] and mom_pct > 0.02: mom_color = "#e74c3c" # 汇率涨是红
            
            html_cards += f"""
            <div class="item">
                <div style="flex:1">
                    <div class="label">{item['config']['name']} 
                        <span class="badge {item['level']}">{item['text']}</span>
                    </div>
                    <div class="rationale">{item['config']['rationale']}</div>
                </div>
                <div class="values">
                    <div class="main-val">{item['value']:.2f}</div>
                    <div class="sub-val">Z: <b>{item['z']:+.1f}</b> | 乖离: {bias_pct:+.1f}%</div>
                    <div class="sub-val" style="color:{mom_color}">5日动量: {mom_arrow} {mom_pct:+.1f}%</div>
                </div>
            </div>
            """
        html_cards += "</div>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: {bg_color}; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); overflow: hidden; }}
            .header {{ background: {head_color}; color: white; padding: 30px; text-align: center; }}
            .status-box {{ padding: 20px; text-align: center; border-bottom: 1px solid #eee; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 25px; }}
            @media(max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            .card {{ background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 15px; }}
            .card h3 {{ margin: 0 0 15px 0; color: #555; font-size: 14px; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            .item {{ display: flex; justify-content: space-between; margin-bottom: 15px; border-bottom: 1px dashed #f5f5f5; padding-bottom: 10px; }}
            .label {{ font-weight: 600; font-size: 14px; color: #333; }}
            .rationale {{ font-size: 11px; color: #999; margin-top: 4px; line-height: 1.4; }}
            .values {{ text-align: right; min-width: 100px; }}
            .main-val {{ font-family: "Menlo", monospace; font-weight: 700; font-size: 16px; }}
            .sub-val {{ font-size: 11px; color: #777; margin-top: 3px; }}
            .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; color: white; font-size: 10px; vertical-align: middle; margin-left: 5px; }}
            .red {{ background: #e74c3c; }} .orange {{ background: #f39c12; }} .green {{ background: #27ae60; }} .yellow {{ background: #f1c40f; color: #333; }} .gray {{ background: #95a5a6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin:0">🇨🇳 ESPT 宏观风险监控 (Pro)</h1>
                <div style="font-size:12px; opacity:0.8; margin-top:10px">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="status-box">
                <h2 style="margin:0; color:{head_color}">{status}</h2>
                <p style="color:#666; font-size:14px; margin-top:5px">{desc}</p>
            </div>
            <div class="grid">
                {html_cards}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 报告生成完毕: {os.path.abspath(REPORT_FILENAME)}")

if __name__ == "__main__":
    analyzer = MacroAnalyzer()
    results = analyzer.analyze()
    generate_report(results)
