import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

class MacroAnalyzer:
    def __init__(self):
        self.window_long = 252  # 1年交易日
        self.min_data_points = int(self.window_long * 1.2)  # 最小数据要求
        self.z_thresholds = {"red": 2.0, "orange": 1.0, "green": -1.0}

    def align_time_series(self, series1, series2):
        """智能对齐两个时间序列 (用于比率分析)"""
        if series1.index.tz: series1.index = series1.index.tz_localize(None)
        if series2.index.tz: series2.index = series2.index.tz_localize(None)
        all_dates = series1.index.union(series2.index).sort_values()
        s1 = series1.reindex(all_dates).ffill()
        s2 = series2.reindex(all_dates).ffill()
        valid_mask = ~(s1.isna() | s2.isna())
        return s1[valid_mask], s2[valid_mask]

    def calculate_robust_z_score(self, series, inverse=False):
        """核心算法：乖离率 Z-Score (Bias Z-Score)"""
        # 放宽最小数据要求到窗口的 80% (约200天)，提高鲁棒性
        min_req = int(self.window_long * 0.8)  # <--- 新增：定义最小周期

        if len(series) < min_req:
            return 0, 0.0
        
        # 1. 计算年线 (修改 min_periods)
        rolling_mean = series.rolling(window=self.window_long, min_periods=min_req).mean() # <--- 修改

        # 2. 计算乖离率 (Bias)
        valid_idx = rolling_mean.index[~rolling_mean.isna()]
        if len(valid_idx) == 0:
            return 0, 0.0

        series_valid = series.loc[valid_idx]
        mean_valid = rolling_mean.loc[valid_idx]
        bias_series = (series_valid / mean_valid) - 1

        # 3. Z-Score 标准化 (修改 min_periods)
        # 这里也需要放宽，否则第二层滚动依然会失败
        bias_mean = bias_series.rolling(window=self.window_long, min_periods=min_req).mean() # <--- 修改
        bias_std = bias_series.rolling(window=self.window_long, min_periods=min_req).std()   # <--- 修改

        last_idx = bias_series.index[-1]
        cur_bias = bias_series.loc[last_idx]

        # 加上安全检查，防止刚开始计算时的空值
        if last_idx not in bias_mean.index or pd.isna(bias_mean.loc[last_idx]):
             return 0, cur_bias

        cur_mean = bias_mean.loc[last_idx]
        cur_std = bias_std.loc[last_idx]

        if pd.isna(cur_std) or cur_std == 0:
            z_score = 0
        else:
            z_score = (cur_bias - cur_mean) / cur_std

        # Winsorizing
        z_score = np.clip(z_score, -4.0, 4.0)

        # 风险方向: inverse=True 表示数值越低越危险(如股价)
        risk_z = -z_score if inverse else z_score

        return risk_z, cur_bias


    def fetch_data_safe(self, ticker, period="5y"):
        """带重试的数据获取"""
        for _ in range(2):
            try:
                df = yf.Ticker(ticker).history(period=period)
                if not df.empty and len(df) > 10:
                    return df['Close']
            except Exception:
                time.sleep(1)
        return pd.Series(dtype=float)

    def fetch_and_analyze(self, name, rationale, ticker=None,
                          inverse=False, is_ratio=False,
                          ratio_num=None, ratio_den=None,
                          fallback_ticker=None):
        try:
            series = None
            display_ticker = ticker

            if is_ratio:
                s_num = self.fetch_data_safe(ratio_num)
                s_den = self.fetch_data_safe(ratio_den)
                if s_num.empty or s_den.empty:
                    raise ValueError(f"比率数据源缺失 {ratio_num}/{ratio_den}")
                s_num_aligned, s_den_aligned = self.align_time_series(s_num, s_den)
                if len(s_num_aligned) < self.min_data_points:
                    raise ValueError("对齐后数据长度不足")
                series = s_num_aligned / s_den_aligned
                display_ticker = f"{ratio_num}/{ratio_den}"
            else:
                series = self.fetch_data_safe(ticker)
                if (series.empty or len(series) < self.min_data_points) and fallback_ticker:
                    print(f"⚠️ [{name}] 主数据源 {ticker} 数据不足，切换备用: {fallback_ticker}")
                    series = self.fetch_data_safe(fallback_ticker)
                    display_ticker = fallback_ticker
                if series.empty or len(series) < self.min_data_points:
                    raise ValueError(f"数据源完全失效: {display_ticker}")

            if series.index.tz:
                series.index = series.index.tz_localize(None)

            current_price = series.iloc[-1]
            z_score, bias = self.calculate_robust_z_score(series, inverse)

            if z_score > self.z_thresholds["red"]:
                level, msg = "red", "极度异常"
            elif z_score > self.z_thresholds["orange"]:
                level, msg = "orange", "显著偏离"
            elif z_score < self.z_thresholds["green"]:
                level, msg = "green", "低位安全"
            else:
                level, msg = "yellow", "处于均值"

            return {
                "name": name, "ticker": display_ticker,
                "current": f"{current_price:.2f}",
                "bias": f"{bias*100:+.1f}%",
                "z": z_score, "level": level, "msg": msg, "rationale": rationale
            }
        except Exception as e:
            return {
                "name": name, "ticker": "Error",
                "current": "-", "bias": "-", "z": 0,
                "level": "gray", "msg": "Error",
                "rationale": f"{rationale} (错误: {str(e)[:20]})"
            }

def get_china_indicators():
    print("🔍 正在扫描中国宏观经济 ESPT 指标 (Final Robust版)...")
    analyzer = MacroAnalyzer()
    indicators = {"E": [], "S": [], "P": [], "T": []}

    # === E: 预期 (Expectation) ===
    indicators["E"].append(analyzer.fetch_and_analyze(
        name="消费降级 (PDD/BABA)",
        rationale="逻辑: 拼多多vs阿里。比率飙升 = 市场确认消费降级逻辑。",
        is_ratio=True, ratio_num="PDD", ratio_den="BABA",
        inverse=False
    ))

    # === S: 结构 (Structure) ===
    indicators["S"].append(analyzer.fetch_and_analyze(
        name="核心资产 (沪深300)", ticker="ASHR", fallback_ticker="FXI",
        rationale="逻辑: 系统性水位。剔除单一行业干扰，看整体贝塔。",
        inverse=True
    ))
    indicators["S"].append(analyzer.fetch_and_analyze(
        name="地产板块", ticker="CHIR", fallback_ticker="2202.HK",
        rationale="逻辑: 地产硬着陆风险。ETF或龙头万科的破位信号。",
        inverse=True
    ))
    indicators["S"].append(analyzer.fetch_and_analyze(
        name="汇率信心 (USDCNY)", ticker="USDCNY=X", fallback_ticker="CNH=X",
        rationale="逻辑: 信心之锚。急速贬值(乖离率飙升) = 资本外流压力。",
        inverse=False
    ))

    # === P: 现实/权力 (Power) ===
    indicators["P"].append(analyzer.fetch_and_analyze(
        name="工业需求 (铜)", ticker="HG=F", fallback_ticker="COPX",
        rationale="逻辑: '铜博士'。铜价暴跌 = 实体经济/基建需求失速。",
        inverse=True
    ))

    # === T: 技术 (Technology) ===
    indicators["T"].append(analyzer.fetch_and_analyze(
        name="科技相对强弱",
        rationale="逻辑: CN科技 vs 美股大盘。剔除全球贝塔，看独立阿尔法风险。",
        is_ratio=True, ratio_num="CQQQ", ratio_den="SPY",
        inverse=True
    ))

    return indicators

def generate_html_report_china(indicators, total_score, final_risk, advice, veto_triggered=False, weights=None):
    """生成中国ESPT指标的HTML报告（优化排版版）"""
    if weights is None:
        weights = {"E": 0.20, "S": 0.35, "P": 0.30, "T": 0.15}

    html_style = """
    <style>
    :root {
        --red: #e74c3c;
        --orange: #e67e22;
        --yellow: #f1c40f;
        --green: #27ae60;
        --gray: #95a5a6;
        --light-bg: #f8f9fa;
        --card-bg: #ffffff;
        --border-color: #e0e0e0;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        background-color: var(--light-bg);
        color: #333;
        line-height: 1.6;
        padding: 16px;
    }
    .dashboard {
        max-width: 1200px;
        margin: 0 auto;
        background: var(--card-bg);
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    .header {
        background: linear-gradient(135deg, #d52b1e 0%, #f8c300 100%);
        color: white;
        padding: 24px 32px;
        text-align: center;
    }
    .header h1 {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        font-size: 28px;
        margin-bottom: 8px;
    }
    .timestamp {
        opacity: 0.9;
        font-size: 14px;
    }

    .risk-summary {
        padding: 24px 32px;
        background-color: #fff9e6;
        border-bottom: 1px solid var(--border-color);
    }
    .risk-title {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 16px;
        color: #d52b1e;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .risk-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
    }
    .risk-box {
        background: white;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        text-align: center;
    }
    .risk-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 6px;
    }
    .risk-value {
        font-size: 22px;
        font-weight: bold;
    }

    .veto-warning {
        background-color: #ffebee;
        border-left: 4px solid var(--red);
        padding: 16px;
        margin: 0 32px 24px;
        border-radius: 8px;
        color: var(--red);
        font-weight: 500;
    }

    .dimension-card {
        padding: 24px 32px;
        border-bottom: 1px solid var(--border-color);
    }
    .dimension-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #2c3e50;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .indicator-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
        gap: 20px;
    }
    .indicator-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .indicator-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 14px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .indicator-name {
        font-weight: bold;
        font-size: 16px;
        flex: 1;
    }
    .ticker {
        font-size: 12px;
        color: #666;
        background-color: #f0f0f0;
        padding: 2px 8px;
        border-radius: 12px;
        white-space: nowrap;
    }
    .indicator-details {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-bottom: 14px;
    }
    .detail-item {
        font-size: 14px;
    }
    .detail-label {
        font-weight: bold;
        color: #666;
        display: inline-block;
        width: 70px;
    }
    .rationale {
        font-size: 13px;
        color: #555;
        line-height: 1.5;
        padding: 12px;
        background-color: #f9f9f9;
        border-radius: 6px;
        border-left: 3px solid #ddd;
    }

    .level-red { color: var(--red); font-weight: bold; }
    .level-orange { color: var(--orange); font-weight: bold; }
    .level-green { color: var(--green); font-weight: bold; }
    .level-yellow { color: var(--yellow); font-weight: bold; }
    .level-gray { color: var(--gray); font-weight: bold; }

    .footer {
        padding: 20px 32px;
        text-align: center;
        color: #7f8c8d;
        font-size: 13px;
        border-top: 1px solid var(--border-color);
    }

    @media (max-width: 768px) {
        .indicator-grid { grid-template-columns: 1fr; }
        .risk-details { grid-template-columns: 1fr; }
        .header h1 { font-size: 22px; }
        .dashboard { border-radius: 12px; }
        body { padding: 12px; }
    }
    </style>
    """

    color_map = {
        "red": "#e74c3c", "orange": "#e67e22",
        "green": "#27ae60", "yellow": "#f1c40f", "gray": "#95a5a6"
    }
    icon_map = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "green": "🟢", "gray": "⚪"}
    dimension_icons = {"E": "📊", "S": "🏛️", "P": "⚖️", "T": "💻"}

    risk_color_key = "gray"
    if "🔴" in final_risk: risk_color_key = "red"
    elif "🟠" in final_risk: risk_color_key = "orange"
    elif "🟡" in final_risk: risk_color_key = "yellow"
    elif "🟢" in final_risk: risk_color_key = "green"

    html_body = "<body>"
    html_body += f"""
    <div class='dashboard'>
        <div class='header'>
            <h1><span>🇨🇳</span>中国宏观经济 ESPT 仪表盘</h1>
            <p class='timestamp'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class='risk-summary'>
            <div class='risk-title'>🛡️ 风险综述</div>
            <div class='risk-details'>
                <div class='risk-box'>
                    <div class='risk-label'>加权风险分</div>
                    <div class='risk-value' style="color: {color_map[risk_color_key]}">{total_score:.2f} / 10.0</div>
                </div>
                <div class='risk-box'>
                    <div class='risk-label'>最终评级</div>
                    <div class='risk-value'>{final_risk}</div>
                </div>
                <div class='risk-box'>
                    <div class='risk-label'>交易建议</div>
                    <div class='risk-value' style="font-size: 16px; word-break: break-word;">{advice[:60]}</div>
                </div>
            </div>
        </div>
    """

    if veto_triggered:
        html_body += """
        <div class='veto-warning'>
            ⚠️ 触发系统性熔断机制：检测到多个核心指标同时异常，建议立即采取防御性策略
        </div>
        """

    for dim, items in indicators.items():
        html_body += f"""
        <div class='dimension-card'>
            <div class='dimension-title'>
                <span>{dimension_icons.get(dim, '📈')}</span>
                {dim} 维度 (权重 {weights.get(dim, 0.25)*100:.0f}%)
            </div>
            <div class='indicator-grid'>
        """
        for item in items:
            level = item.get('level', 'gray')
            z_val = item.get('z', 0)
            z_str = f"{z_val:+.2f}σ"
            html_body += f"""
            <div class='indicator-card'>
                <div class='indicator-header'>
                    <div class='indicator-name'>{icon_map.get(level, '⚪')} {item.get('name', 'N/A')}</div>
                    <div class='ticker'>{item.get('ticker', 'N/A')}</div>
                </div>
                <div class='indicator-details'>
                    <div class='detail-item'><span class='detail-label'>当前值:</span> {item.get('current', '-')}</div>
                    <div class='detail-item'><span class='detail-label'>乖离率:</span> {item.get('bias', '-')}</div>
                    <div class='detail-item'><span class='detail-label'>Z-Score:</span> <span class='level-{level}'>{z_str}</span></div>
                    <div class='detail-item'><span class='detail-label'>状态:</span> <span class='level-{level}'>{item.get('msg', '-')}</span></div>
                </div>
                <div class='rationale'>{item.get('rationale', '无原理解读')}</div>
            </div>
            """
        html_body += "</div></div>"

    html_body += """
        <div class='footer'>
            <p>ESPT分析框架 | 中国宏观经济仪表盘 | 基于乖离率Z-Score算法</p>
            <p>免责声明：本报告仅供参考，不构成任何投资建议</p>
        </div>
    </div>
    </body>
    """

    final_html = f"""<!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>中国宏观经济 ESPT 仪表盘</title>
        {html_style}
    </head>
    {html_body}
    </html>"""

    filename = "china_econ_report.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"✅ 生成HTML报告: {filename}")
    return filename

def generate_report(indicators):
    print("\n" + "="*95)
    print("🇨🇳 中国宏观经济 ESPT 仪表盘 (乖离率算法 + 智能备用版)")
    print("="*95)
    weights = {"E": 0.20, "S": 0.35, "P": 0.30, "T": 0.15}
    score_map = {"red": 10, "orange": 6, "yellow": 3, "green": 0, "gray": 5}
    total_score = 0
    veto_triggered = False
    veto_msgs = []
    insights = []
    core_status = {}

    for dim, items in indicators.items():
        print(f"\n【{dim} 维度】 (权重 {weights[dim]*100:.0f}%)")
        print(f"  {'指标名称':<22} | {'Z-Score':<8} | {'Bias(乖离)':<10} | {'状态':<8} | {'原理解读'}")
        print("  " + "-"*90)
        dim_score = 0
        for item in items:
            dim_score += score_map.get(item['level'], 0)
            icon = {"red":"🔴","orange":"🟠","yellow":"🟡","green":"🟢","gray":"⚪"}.get(item['level'], "⚪")
            z_val = item.get('z', 0)
            z_str = f"{z_val:+.2f}σ"
            print(f"  {icon} {item['name']:<20} | {z_str:<8} | {item['bias']:<10} | {item['msg']:<8} | {item['rationale'][:28]}...")
            if abs(z_val) > 1.5:
                insights.append(f"👉 [{item['name']}] 信号显著: Z={z_str}, 乖离率={item['bias']}。")
            if "沪深300" in item['name']: core_status['ASHR'] = item['level']
            if "USDCNY" in item['name']: core_status['FX'] = item['level']
            if "铜" in item['name']: core_status['Copper'] = item['level']
        total_score += (dim_score / len(items)) * weights[dim]

    if core_status.get('FX') == 'red':
        veto_msgs.append("汇率失锚 (USDCNY Red)")
    if core_status.get('ASHR') == 'red':
        veto_msgs.append("核心资产崩盘 (ASHR Red)")
    orange_cnt = sum(1 for status in core_status.values() if status == 'orange')
    if orange_cnt >= 2:
        veto_msgs.append(f"系统性共振 ({orange_cnt}个核心指标 Orange)")
    if veto_msgs:
        veto_triggered = True

    print("\n" + "="*95)
    print("🧠 深度逻辑透视 (Deep Dive Analysis)")
    print("-" * 95)
    if not insights:
        print("  当前各项宏观代理指标运行平稳，乖离率在正常区间内。")
    else:
        for insight in insights:
            print(insight)

    print("\n" + "="*95)
    print("🛡️ 风险综述与交易建议")
    print("-" * 95)
    if veto_triggered:
        final_risk = "🔴 红色 (系统性熔断)"
        reason = " + ".join(veto_msgs)
        advice = f"触发熔断机制: [{reason}]。建议清仓观望，持有美元/黄金/国债。"
    elif total_score > 6:
        final_risk = "🟠 橙色 (高压警戒)"
        advice = "宏观环境显著恶化，乖离率偏离过大。建议大幅降低权益仓位，仅保留高股息防守。"
    elif total_score > 3:
        final_risk = "🟡 黄色 (震荡/结构性)"
        advice = "市场缺乏明确宏观方向。轻指数，重结构（关注科技相对强弱或消费降级逻辑）。"
    else:
        final_risk = "🟢 绿色 (安全/复苏)"
        advice = "宏观指标健康或处于超跌反弹区。适合右侧布局顺周期资产 (ASHR/Copper)。"

    print(f"📊 加权风险分: {total_score:.2f} / 10.0")
    print(f"🏁 最终评级: {final_risk}")
    print(f"💡 交易建议: {advice}")
    print("="*95)

    html_file = generate_html_report_china(indicators, total_score, final_risk, advice, veto_triggered, weights)
    return {
        "total_score": total_score,
        "final_risk": final_risk,
        "advice": advice,
        "veto_triggered": veto_triggered,
        "html_file": html_file
    }

if __name__ == "__main__":
    try:
        data = get_china_indicators()
        result = generate_report(data)
        print(f"\n📄 HTML报告已保存至: {result['html_file']}")
    except Exception as e:
        print(f"Critical Error: {e}")




