import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==========================================
# 1. 核心量化引擎 (实时直连版)
# ==========================================
class MacroAnalyzer:
    def __init__(self):
        self.window_long = 252  # 1年交易日
        self.min_data_points = int(self.window_long * 0.85)

    def fetch_batch_data(self, tickers_dict, period="2y"):
        """
        🚀 实时获取数据 (无缓存模式)
        每次运行都强制从网络下载最新数据。
        """
        # 1. 提取所有 Ticker 并去重
        all_tickers = list(set([t for val in tickers_dict.values() for t in (val if isinstance(val, list) else [val])]))
        print(f"🌐 [Network] 正在请求实时数据 ({len(all_tickers)} 个标的)...")
        
        try:
            # group_by='ticker' 确保数据结构清晰
            df = yf.download(all_tickers, period=period, group_by='ticker', auto_adjust=False, threads=True)
            if df.empty:
                print("⚠️ 警告: 下载的数据为空，请检查网络连接。")
            return df
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return pd.DataFrame()

    def extract_series(self, df_batch, ticker):
        """安全提取单个序列"""
        try:
            # 兼容 yfinance 的多层索引结构
            if ticker in df_batch.columns.levels[0]:
                data = df_batch[ticker]
                # 优先 Close, 其次 Adj Close
                s = data['Close'] if 'Close' in data.columns else data.get('Adj Close', pd.Series(dtype=float))
                # 清洗数据：去0，去空，移除时区
                s = s.replace(0, np.nan).dropna()
                if s.index.tz: s.index = s.index.tz_localize(None)
                return s
        except: pass
        return pd.Series(dtype=float)

    def compute_synthetic_index(self, series_list, operation="product"):
        """
        🔧 核心修复：跨市场数据对齐引擎
        解决美股/日股休市日不一致导致的数据断裂问题。
        """
        if not series_list: return pd.Series(dtype=float)
        
        # 1. 取所有日期的并集
        all_dates = series_list[0].index
        for s in series_list[1:]: all_dates = all_dates.union(s.index)
        all_dates = all_dates.sort_values()
        
        # 2. 前向填充 (FFill): 如果今天某市场休市，沿用昨天价格
        aligned = [s.reindex(all_dates).ffill() for s in series_list]
        
        # 3. 向量化计算
        result = aligned[0]
        if operation == "product": # 乘法 (如 痛苦指数)
            for s in aligned[1:]: result = result * s
        elif operation == "sum":   # 加法 (如 巴菲特篮子)
            for s in aligned[1:]: result = result + s
            
        return result.dropna()

    def generate_sparkline(self, series, days=30):
        """🎨 生成 SVG 微型走势图"""
        if len(series) < days: return ""
        # 取最近N天数据
        data = series.iloc[-days:].values
        min_val, max_val = np.min(data), np.max(data)
        if max_val == min_val: return ""
        
        points = []
        width, height = 100, 30
        step = width / (days - 1)
        
        for i, val in enumerate(data):
            x = i * step
            # SVG坐标系翻转 (y=0在顶部)
            y = height - ((val - min_val) / (max_val - min_val) * height)
            points.append(f"{x:.1f},{y:.1f}")
            
        color = "#ef4444" if data[-1] < data[0] else "#10b981" # 跌红涨绿
        return f'<svg width="{width}" height="{height}"><polyline points="{" ".join(points)}" style="fill:none;stroke:{color};stroke-width:1.5" /></svg>'

    def analyze_item(self, name, series, risk_type="high_is_risk", desc=""):
        """
        ⚖️ 核心评级逻辑 (正统 Z-Score)
        正数 = 高于均线 (Up Trend)
        负数 = 低于均线 (Down Trend)
        """
        if series.empty or len(series) < self.min_data_points:
            return {"name": name, "level": "gray", "text": "数据不足", "z": 0, "pct": 0, "spark": ""}

        # 1. 计算均线与乖离率
        ma252 = series.rolling(window=self.window_long).mean()
        bias = (series / ma252) - 1
        
        # 2. 计算 Z-Score (不人工取反，保持统计真实性)
        bias_mean = bias.rolling(window=self.window_long).mean()
        bias_std = bias.rolling(window=self.window_long).std()
        
        cur_val = series.iloc[-1]
        cur_bias = bias.iloc[-1]
        
        if pd.isna(bias_std.iloc[-1]) or bias_std.iloc[-1] == 0: z = 0
        else: z = (cur_bias - bias_mean.iloc[-1]) / bias_std.iloc[-1]
        z = np.clip(z, -4.5, 4.5)

        # 3. 计算历史百分位 (Rank) - 衡量当前位置在过去一年的极端程度
        recent_bias = bias.iloc[-self.window_long:]
        pct_rank = (recent_bias < cur_bias).mean() * 100

        # 4. 颜色评级判断 (根据业务类型)
        level, text = "blue", "正常"
        
        # A: 越高越危险 (如: VIX, 痛苦指数)
        if risk_type == "high_is_risk":
            if z > 2.2:      level, text = "red", "极度过热 ⚠️"
            elif z > 1.25:   level, text = "orange", "风险上升"
            elif z < -1.0:   level, text = "green", "低位安全"
            
        # B: 越低越危险 (如: 股市, 经济数据)
        elif risk_type == "low_is_risk":
            if z < -2.2:     level, text = "red", "崩盘/枯竭 ⚠️"
            elif z < -1.25:  level, text = "orange", "显著回调"
            elif z > 1.5:    level, text = "green", "趋势强劲"
            
        # C: 双向风险 (如: 汇率)
        elif risk_type == "two_sided":
            if z > 2.5:      level, text = "red", "失控贬值 (干预)"
            elif z > 1.0:    level, text = "green", "有利贬值"
            elif z < -2.0:   level, text = "red", "暴力升值 (崩盘)"

        spark = self.generate_sparkline(series)

        return {
            "name": name, "value": cur_val, "z": z, "bias": cur_bias, 
            "pct": pct_rank, "level": level, "text": text, 
            "desc": desc, "spark": spark
        }

# ==========================================
# 2. 业务配置 (Japan Config)
# ==========================================
def get_japan_dashboard():
    analyzer = MacroAnalyzer()
    
    # 定义需要的代码
    config = {
        "N225": "^N225",           # 日经225
        "Oil": "CL=F",             # WTI原油
        "Yen": "USDJPY=X",         # 美元兑日元
        "Banks": "8306.T",         # 三菱日联 (加息代理)
        "REITs": "1343.T",         # 东证REITs (资产通胀)
        "Semi": "8035.T",          # 东京电子 (科技Beta)
        "TLT": "TLT",              # 20年美债 (外部压力)
        "Buffett": ["8058.T", "8031.T", "8001.T", "8002.T", "8053.T"] # 五大商社
    }
    
    # 1. 实时获取
    df = analyzer.fetch_batch_data(config)
    dashboard = {"宏观脉搏 (Macro)": [], "市场结构 (Structure)": [], "主力资金 (Flow)": []}

    # 提取 Series
    s_oil = analyzer.extract_series(df, "CL=F")
    s_yen = analyzer.extract_series(df, "USDJPY=X")
    s_bank = analyzer.extract_series(df, "8306.T")
    s_reit = analyzer.extract_series(df, "1343.T")
    s_tlt = analyzer.extract_series(df, "TLT")

    # --- 组合指标逻辑 ---
    
    # 1. 家庭痛苦指数 (Oil * Yen)
    s_pain = analyzer.compute_synthetic_index([s_oil, s_yen], "product")
    dashboard["宏观脉搏 (Macro)"].append(analyzer.analyze_item(
        "家庭痛苦指数", s_pain, "high_is_risk", 
        "逻辑: 油价×汇率。Z为正 = 输入性通胀压力大。"
    ))
    
    # 2. 汇率双向风险
    dashboard["宏观脉搏 (Macro)"].append(analyzer.analyze_item(
        "日元汇率 (USD/JPY)", s_yen, "two_sided",
        "逻辑: Z>2.5 警戒央行干预; Z<-2.0 警戒套息平仓。"
    ))

    # 3. 结构性指标
    dashboard["市场结构 (Structure)"].append(analyzer.analyze_item(
        "加息押注 (MUFG)", s_bank, "high_is_risk", 
        "逻辑: 银行暴涨(Z正) = 押注YCC取消 = 债市利空。"
    ))
    
    dashboard["市场结构 (Structure)"].append(analyzer.analyze_item(
        "资产通胀 (J-REIT)", s_reit, "low_is_risk",
        "逻辑: 地产信托。Z为负代表通缩回归，利空。"
    ))

    # 4. 巴菲特篮子 (Sum of 5 Stocks)
    buffett_list = [analyzer.extract_series(df, t) for t in config["Buffett"]]
    s_buffett = analyzer.compute_synthetic_index(buffett_list, "sum")
    dashboard["主力资金 (Flow)"].append(analyzer.analyze_item(
        "巴菲特五大商社", s_buffett, "low_is_risk",
        "逻辑: 外资核心配置。Z为正代表外资流入强劲。"
    ))
    
    # 5. 外部利率压力 (TLT)
    dashboard["主力资金 (Flow)"].append(analyzer.analyze_item(
        "外部利率压力 (TLT)", s_tlt, "low_is_risk",
        "逻辑: Z为负(暴跌)代表美债利率飙升，日央行压力剧增。"
    ))

    return dashboard

# ==========================================
# 3. 报告可视化 (HTML Generator)
# ==========================================
def generate_html(dashboard):
    # 熔断判定逻辑
    st = {item['name']: item['level'] for cat in dashboard.values() for item in cat}
    
    overall_title = "🟢 市场环境：温和 (Neutral)"
    header_bg = "linear-gradient(135deg, #10b981 0%, #059669 100%)" # Green
    
    veto_triggers = []
    if st.get('家庭痛苦指数') == 'red' and st.get('加息押注 (MUFG)') == 'red':
        veto_triggers.append("滞胀双杀 (Stagflation)")
    if st.get('日元汇率 (USD/JPY)') == 'red':
        veto_triggers.append("汇率失控 (FX Crisis)")
    if st.get('外部利率压力 (TLT)') == 'red':
        veto_triggers.append("美债风暴 (Rates Shock)")

    if veto_triggers:
        overall_title = f"🔴 极度风险：{' + '.join(veto_triggers)}"
        header_bg = "linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)" # Red

    html = f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>Japan Real-Time Sentinel</title>
        <style>
            :root {{ --bg: #f8fafc; --card: #ffffff; --text: #334155; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 40px; margin: 0; }}
            .container {{ max-width: 960px; margin: 0 auto; }}
            
            .header {{ background: {header_bg}; color: white; padding: 35px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            .header h1 {{ margin: 0; font-size: 26px; }}
            .meta {{ font-size: 14px; opacity: 0.9; margin-top: 10px; font-family: monospace; }}
            
            .section-title {{ font-size: 16px; font-weight: bold; color: #64748b; margin: 25px 0 10px 5px; border-left: 4px solid #cbd5e1; padding-left: 10px; }}
            
            .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); overflow: hidden; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: left; padding: 12px 20px; background: #f1f5f9; color: #64748b; font-size: 12px; font-weight: 600; }}
            td {{ padding: 12px 20px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }}
            tr:last-child td {{ border-bottom: none; }}
            
            .name {{ font-weight: bold; font-size: 14px; display: block; }}
            .desc {{ font-size: 11px; color: #94a3b8; }}
            
            .tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white; }}
            .red {{ background: #ef4444; }} .orange {{ background: #f97316; }} 
            .green {{ background: #10b981; }} .blue {{ background: #3b82f6; }} .gray {{ background: #94a3b8; }}
            
            .z-val {{ font-family: monospace; font-weight: bold; font-size: 13px; }}
            .rank-val {{ font-size: 10px; color: #64748b; }}
            
            .footer {{ text-align: center; margin-top: 40px; font-size: 11px; color: #cbd5e1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🇯🇵 Japan Sentinel <span style="font-size:16px; opacity:0.8;">| 实时宏观监测</span></h1>
                <div class="meta">{overall_title}</div>
                <div class="meta">数据更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
    """
    
    for cat, items in dashboard.items():
        html += f"<div class='section-title'>{cat}</div><div class='card'><table>"
        html += "<thead><tr><th width='35%'>指标</th><th width='15%'>状态</th><th width='20%'>数据 (Z | Rank)</th><th width='30%'>30日趋势</th></tr></thead><tbody>"
        
        for item in items:
            html += f"""
            <tr>
                <td>
                    <span class="name">{item['name']}</span>
                    <span class="desc">{item['desc']}</span>
                </td>
                <td><span class="tag {item['level']}">{item['text']}</span></td>
                <td>
                    <div class="z-val">Z: {item['z']:+.2f}</div>
                    <div class="rank-val">Rank: {item['pct']:.0f}%</div>
                </td>
                <td>{item['spark']}</td>
            </tr>
            """
        html += "</tbody></table></div>"
        
    html += """
            <div class="footer">
                Algorithm: Standard Z-Score (Window: 252) | Data Source: Yahoo Finance Real-time
            </div>
        </div>
    </body>
    </html>
    """
    
    filename = "japan_espt_optimized.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 报告生成完毕: {os.path.abspath(filename)}")

if __name__ == "__main__":
    try:
        data = get_japan_dashboard()
        generate_html(data)
    except KeyboardInterrupt:
        print("\n用户停止。")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
