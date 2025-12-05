import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time


class MacroAnalyzer:
    def __init__(self):
        self.window_long = 252  # 1年交易日
        self.min_data_points = int(self.window_long * 1.2)
        self.z_thresholds = {"red": 2.0, "orange": 1.0, "green": -1.0}

    def align_time_series(self, series1, series2):
        """智能对齐两个时间序列 (用于合成 Pain Index)"""
        if series1.index.tz:
            series1.index = series1.index.tz_localize(None)
        if series2.index.tz:
            series2.index = series2.index.tz_localize(None)

        all_dates = series1.index.union(series2.index).sort_values()
        s1 = series1.reindex(all_dates).ffill()
        s2 = series2.reindex(all_dates).ffill()

        valid_mask = ~(s1.isna() | s2.isna())
        return s1[valid_mask], s2[valid_mask]

    def calculate_robust_z_score(self, series, inverse=False):
        """核心算法：乖离率 Z-Score (Bias Z-Score)"""
        if len(series) < self.min_data_points:
            return 0, 0.0
        # 1. 计算年线 (Rolling Mean)
        rolling_mean = series.rolling(window=self.window_long, min_periods=self.window_long).mean()

        # 2. 计算乖离率 (Bias)
        valid_idx = rolling_mean.index[~rolling_mean.isna()]
        if len(valid_idx) == 0:
            return 0, 0.0

        series_valid = series.loc[valid_idx]
        mean_valid = rolling_mean.loc[valid_idx]
        bias_series = (series_valid / mean_valid) - 1

        # 3. Z-Score 标准化
        bias_mean = bias_series.rolling(window=self.window_long).mean()
        bias_std = bias_series.rolling(window=self.window_long).std()

        last_idx = bias_series.index[-1]
        cur_bias = bias_series.loc[last_idx]

        # 获取最新的统计分布
        cur_mean = bias_mean.loc[last_idx]
        cur_std = bias_std.loc[last_idx]

        if pd.isna(cur_std) or cur_std == 0:
            z_score = 0
        else:
            z_score = (cur_bias - cur_mean) / cur_std

        # Winsorizing & 风险方向
        z_score = np.clip(z_score, -4.0, 4.0)
        risk_z = -z_score if inverse else z_score

        return risk_z, cur_bias

    def fetch_data_safe(self, ticker, period="2y"):
        """带重试的数据获取"""
        for _ in range(2):
            try:
                df = yf.Ticker(ticker).history(period=period)
                if not df.empty and len(df) > 10:
                    return df['Close']
            except:
                time.sleep(1)
        return pd.Series(dtype=float)

    def fetch_and_analyze(self, name, rationale, ticker=None,
                          inverse=False, external_series=None,
                          fallback_ticker=None):
        try:
            series = None
            display_ticker = ticker

            # --- 模式A: 外部合成序列 (如 Pain Index) ---
            if external_series is not None:
                series = external_series
                display_ticker = "Composite"

            # --- 模式B: 标准 Ticker ---
            else:
                series = self.fetch_data_safe(ticker)
                # 备用机制
                if (series.empty or len(series) < self.min_data_points) and fallback_ticker:
                    print(f"⚠️ [{name}] 主代码 {ticker} 无效，切换备用: {fallback_ticker}")
                    series = self.fetch_data_safe(fallback_ticker)
                    display_ticker = fallback_ticker

                if series.empty:
                    raise ValueError("数据源完全失效")
            # 时区清洗 - 这行要放在try块内部，修复缩进问题
            if series.index.tz:
                series.index = series.index.tz_localize(None)

            # 计算
            current_val = series.iloc[-1]
            z_score, bias = self.calculate_robust_z_score(series, inverse)

            # 评级
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
                "current": f"{current_val:.2f}",
                "bias": f"{bias * 100:+.1f}%",
                "z": z_score, "level": level, "msg": msg, "rationale": rationale
            }
        except Exception as e:
            return {"name": name, "ticker": "Error", "current": "-", "bias": "-", "z": 0, "level": "gray",
                    "msg": "Error", "rationale": str(e)}


analyzer = MacroAnalyzer()


def get_japan_indicators():
    print("🔍 正在扫描日本宏观经济 ESPT 指标 (Japan Robust Ver)...")
    indicators = {"E": [], "S": [], "P": [], "T": []}
    # === E: 预期 (Expectation) ===

    # 1. 输入性通胀 (Pain Index)
    try:
        oil = analyzer.fetch_data_safe("CL=F")
        yen = analyzer.fetch_data_safe("USDJPY=X")

        if not oil.empty and not yen.empty:
            oil, yen = analyzer.align_time_series(oil, yen)
            pain_index = oil * yen

            indicators["E"].append(analyzer.fetch_and_analyze(
                name="输入通胀 (Pain Index)",
                ticker="Oil*Yen",
                rationale="逻辑: 原油x日元。双高代表进口成本爆炸，家庭实际购买力剧减。",
                external_series=pain_index,
                inverse=False  # 指数越高，通胀痛苦越大 -> 风险高
            ))
        else:
            raise ValueError("基础数据缺失")
    except Exception as e:
        indicators["E"].append({"name": "输入通胀", "level": "gray", "msg": "Error", "rationale": "计算失败"})
    # 2. 通缩心态 (优衣库 / 迅销)
    indicators["E"].append(analyzer.fetch_and_analyze(
        name="通缩心态 (优衣库)", ticker="9983.T", fallback_ticker="FRCOY",
        rationale="逻辑: 9983.T。股价相对于均线飙升 = 市场确认'消费降级'逻辑固化。",
        inverse=False
    ))
    # === S: 结构 (Structure) ===

    # 1. 利率冲击 (三菱日联)
    indicators["S"].append(analyzer.fetch_and_analyze(
        name="YCC博弈 (三菱日联)", ticker="8306.T", fallback_ticker="MUFG",
        rationale="逻辑: 银行股大涨 = 押注央行加息/YCC取消。对债市是系统性冲击。",
        inverse=False
    ))

    # 2. 资本开支 (Fanuc)
    indicators["S"].append(analyzer.fetch_and_analyze(
        name="全球Capex (Fanuc)", ticker="6954.T", fallback_ticker="FANUY",
        rationale="逻辑: 工业机器人。股价反映全球制造业资本开支周期。",
        inverse=True
    ))
    # === P: 权力/政策 (Power) ===

    # 1. 汇率锚 (USDJPY)
    indicators["P"].append(analyzer.fetch_and_analyze(
        name="汇率风险 (USDJPY)", ticker="USDJPY=X",
        rationale="逻辑: 日本央行的底线。汇率过高(贬值)触发干预风险。",
        inverse=False
    ))
    # 2. 地缘/军工 (三菱重工)
    indicators["P"].append(analyzer.fetch_and_analyze(
        name="地缘风险 (三菱重工)", ticker="7011.T", fallback_ticker="MHVYf",
        rationale="逻辑: 军工股。反映东亚地缘政治紧张度。",
        inverse=False
    ))
    # === T: 技术 (Technology) ===

    # 1. 半导体上游 (东京电子 TEL)
    indicators["T"].append(analyzer.fetch_and_analyze(
        name="AI周期 (东京电子)", ticker="8035.T", fallback_ticker="TOELY",
        rationale="逻辑: 半导体设备。日本掌握上游核心，反映全球AI硬件需求。",
        inverse=True
    ))
    return indicators


def generate_html_report_japan(indicators, total_score, final_risk, advice, veto_triggered=False, veto_msgs=None):
    """生成日本ESPT指标的HTML报告"""

    html_style = """
    <style>
        body { font-family: "Segoe UI", "Hiragino Sans", "Meiryo", sans-serif; background-color: #f9f7f7; padding: 20px; color: #333; }
        .dashboard { max-width: 1000px; margin: auto; background-color: white; border-radius: 10px; box-shadow: 0 5px 25px rgba(0,0,0,0.08); overflow: hidden; border: 1px solid #e0e0e0; }
        .header { background: linear-gradient(135deg, #bc002d 0%, #f5878c 100%); color: white; padding: 25px 30px; text-align: center; }
        .header h1 { margin: 0 0 10px 0; font-size: 28px; display: flex; align-items: center; justify-content: center; }
        .flag-icon { font-size: 34px; margin-right: 15px; }
        .timestamp { font-size: 14px; opacity: 0.9; }
        .risk-summary { padding: 25px 30px; background-color: #fff5f5; border-bottom: 1px solid #ffd9d9; }
        .risk-title { font-size: 20px; font-weight: bold; margin-bottom: 15px; color: #bc002d; display: flex; align-items: center; }
        .risk-title-icon { margin-right: 10px; }
        .risk-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .risk-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 3px 10px rgba(0,0,0,0.05); text-align: center; border-top: 4px solid #bc002d; }
        .risk-label { font-weight: bold; color: #666; margin-bottom: 8px; font-size: 14px; }
        .risk-value { font-size: 26px; font-weight: bold; }
        .dimension-section { padding: 25px 30px; border-bottom: 1px solid #eee; }
        .dimension-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .dimension-title { font-size: 22px; font-weight: bold; color: #2c3e50; }
        .dimension-subtitle { font-size: 14px; color: #7f8c8d; }
        .indicators-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 20px; }
        .indicator-card { padding: 18px; border-radius: 8px; background-color: #f8f9fa; border-left: 4px solid #ddd; position: relative; }
        .indicator-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .indicator-name { font-weight: bold; font-size: 16px; }
        .indicator-ticker { font-size: 12px; color: #666; background-color: #eee; padding: 2px 8px; border-radius: 10px; }
        .indicator-metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 12px; }
        .metric-item { font-size: 14px; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { font-weight: 600; }
        .indicator-rationale { font-size: 13px; color: #555; line-height: 1.5; padding: 12px; background-color: white; border-radius: 6px; margin-top: 8px; border-left: 3px solid #ddd; }
        .level-red { color: #e74c3c; font-weight: bold; }
        .level-orange { color: #e67e22; font-weight: bold; }
        .level-green { color: #27ae60; font-weight: bold; }
        .level-yellow { color: #f1c40f; font-weight: bold; }
        .level-gray { color: #95a5a6; font-weight: bold; }
        .veto-section { background-color: #ffe6e6; border-left: 5px solid #e74c3c; padding: 20px; margin: 20px 30px; border-radius: 8px; }
        .veto-title { color: #e74c3c; font-weight: bold; margin-bottom: 10px; font-size: 18px; display: flex; align-items: center; }
        .veto-icon { margin-right: 10px; }
        .footer { padding: 20px 30px; text-align: center; color: #7f8c8d; font-size: 13px; border-top: 1px solid #eee; background-color: #f9f7f7; }
        .methodology { font-size: 12px; color: #95a5a6; margin-top: 10px; }
    </style>
    """

    # 颜色和图标映射
    color_map = {
        "red": "#e74c3c", "orange": "#e67e22",
        "green": "#27ae60", "yellow": "#f1c40f", "gray": "#95a5a6"
    }

    icon_map = {
        "red": "🔴", "orange": "🟠", "yellow": "🟡",
        "green": "🟢", "gray": "⚪"
    }

    dimension_titles = {
        "E": "预期 (Expectation)",
        "S": "结构 (Structure)",
        "P": "权力/政策 (Power)",
        "T": "技术 (Technology)"
    }

    dimension_icons = {"E": "📊", "S": "🏛️", "P": "⚖️", "T": "💻"}

    # 确定风险等级对应的颜色
    risk_color = "green"
    if "🔴" in final_risk:
        risk_color = "red"
    elif "🟠" in final_risk:
        risk_color = "orange"
    elif "🟡" in final_risk:
        risk_color = "yellow"

    html_body = "<body>"
    html_body += f"""
    <div class='dashboard'>
        <div class='header'>
            <h1><span class='flag-icon'>🇯🇵</span>日本宏观经济 ESPT 仪表盘</h1>
            <p class='timestamp'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Pain Index + Bias Z-Score算法</p>
        </div>
        
        <div class='risk-summary'>
            <div class='risk-title'>
                <span class='risk-title-icon'>📋</span> 风险综合评估
            </div>
            <div class='risk-grid'>
                <div class='risk-card'>
                    <div class='risk-label'>加权风险评分</div>
                    <div class='risk-value' style="color: {color_map.get(risk_color, '#bc002d')}">
                        {total_score:.2f} / 10.0
                    </div>
                </div>
                <div class='risk-card'>
                    <div class='risk-label'>宏观风险评级</div>
                    <div class='risk-value'>{final_risk}</div>
                </div>
                <div class='risk-card'>
                    <div class='risk-label'>核心交易策略</div>
                    <div class='risk-value' style="font-size: 18px; color: #2c3e50;">{advice[:45]}...</div>
                </div>
            </div>
        </div>
    """

    if veto_triggered and veto_msgs:
        html_body += f"""
        <div class='veto-section'>
            <div class='veto-title'>
                <span class='veto-icon'>⚠️</span> 复合熔断机制触发
            </div>
            <div><strong>触发条件:</strong> {' + '.join(veto_msgs)}</div>
            <div style="margin-top: 8px; font-size: 14px;">建议立即采取防御性策略，规避系统性风险</div>
        </div>
        """

    # 各维度指标展示
    weights = {"E": 0.20, "S": 0.30, "P": 0.35, "T": 0.15}

    for dim, items in indicators.items():
        dim_title = dimension_titles.get(dim, dim)
        dim_weight = weights.get(dim, 0.25)

        html_body += f"""
        <div class='dimension-section'>
            <div class='dimension-header'>
                <div>
                    <div class='dimension-title'>{dimension_icons.get(dim, '📈')} {dim_title}</div>
                    <div class='dimension-subtitle'>ESPT分析框架 - 权重: {dim_weight*100:.0f}%</div>
                </div>
            </div>
            
            <div class='indicators-container'>
        """

        for item in items:
            level = item.get('level', 'gray')
            icon = icon_map.get(level, '⚪')
            border_color = color_map.get(level, '#95a5a6')

            html_body += f"""
            <div class='indicator-card' style="border-left-color: {border_color};">
                <div class='indicator-header'>
                    <div class='indicator-name'>{icon} {item.get('name', 'N/A')}</div>
                    <div class='indicator-ticker'>{item.get('ticker', 'N/A')}</div>
                </div>
                
                <div class='indicator-metrics'>
                    <div class='metric-item'>
                        <span class='metric-label'>当前值:</span>
                        <span class='metric-value'>{item.get('current', '-')}</span>
                    </div>
                    <div class='metric-item'>
                        <span class='metric-label'>乖离率:</span>
                        <span class='metric-value'>{item.get('bias', '-')}</span>
                    </div>
                    <div class='metric-item'>
                        <span class='metric-label'>Z-Score:</span>
                        <span class='metric-value level-{level}'>{item.get('z', 0):+.2f}σ</span>
                    </div>
                    <div class='metric-item'>
                        <span class='metric-label'>状态:</span>
                        <span class='metric-value level-{level}'>{item.get('msg', '-')}</span>
                    </div>
                </div>
                
                <div class='indicator-rationale'>
                    <strong>逻辑解读:</strong> {item.get('rationale', '无原理解读')}
                </div>
            </div>
            """

        html_body += """
            </div>
        </div>
        """

    html_body += """
        <div class='footer'>
            <p>🇯🇵 日本宏观经济 ESPT 分析系统 | 基于乖离率Z-Score算法与复合熔断机制</p>
            <div class='methodology'>
                方法论: ESPT框架 (预期/结构/权力/技术) + Pain Index + 复合熔断逻辑
            </div>
            <p style="margin-top: 15px; font-size: 12px; color: #bdc3c7;">
                免责声明: 本报告仅供参考，不构成任何投资建议。市场有风险，投资需谨慎。
            </p>
        </div>
    </div>
    </body>
    """

    final_html = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>日本宏观经济 ESPT 仪表盘</title>
        {html_style}
    </head>
    {html_body}
    </html>
    """

    filename = f"japan_macro_espt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"\n✅ 生成HTML报告: {filename}")
    return filename


def generate_report(indicators):
    print("\n" + "=" * 95)
    print("🇯🇵 日本宏观经济 ESPT 仪表盘 (Pain Index + Bias Z-Score)")
    print("=" * 95)
    # 日本权重：汇率(P)和结构(S)是核心
    weights = {"E": 0.20, "S": 0.30, "P": 0.35, "T": 0.15}
    score_map = {"red": 10, "orange": 6, "yellow": 3, "green": 0, "gray": 5}

    total_score = 0
    veto_msgs = []
    insights = []

    # 关键状态追踪
    status_tracker = {}
    for dim, items in indicators.items():
        print(f"\n【{dim} 维度】 (权重 {weights[dim] * 100:.0f}%)")
        print(f"  {'指标名称':<20} | {'Z-Score':<8} | {'Bias(乖离)':<10} | {'状态':<8} | {'原理解读'}")
        print("  " + "-" * 90)

        dim_score = 0
        for item in items:
            dim_score += score_map.get(item['level'], 0)
            icon = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "green": "🟢", "gray": "⚪"}.get(item['level'], "⚪")
            z_val = item.get('z', 0)

            print(f"  {icon} {item['name']:<18} | {f'{z_val:+.2f}σ':<8} | {item['bias']:<10} | {item['msg']:<8} | {item['rationale'][:28]}...")
            if abs(z_val) > 1.5:
                insights.append(f"👉 [{item['name']}] 信号显著: Z={z_val:+.2f}σ, 乖离={item['bias']}。")

            # 记录状态用于熔断
            if "Pain Index" in item['name']:
                status_tracker['Pain'] = item['level']
            if "USDJPY" in item['name']:
                status_tracker['Yen'] = item['level']
            if "YCC" in item['name']:
                status_tracker['Bond'] = item['level']
        total_score += (dim_score / len(items)) * weights[dim]
    # === 复合熔断逻辑 (Composite Veto) ===
    # 日本的死穴是：汇率崩盘 OR (输入性通胀爆炸 + 债市崩盘)

    if status_tracker.get('Yen') == 'red':
        veto_msgs.append("汇率失控 (USDJPY Red)")

    if status_tracker.get('Pain') == 'red' and status_tracker.get('Bond') == 'red':
        veto_msgs.append("滞胀+利率双杀 (Pain & YCC Red)")
    veto_triggered = len(veto_msgs) > 0
    print("\n" + "=" * 95)
    print("🧠 深度逻辑透视 (Deep Dive Analysis)")
    print("-" * 95)
    if not insights:
        print("  市场运行平稳，未检测到偏离年线趋势的显著异常。")
    else:
        for insight in insights:
            print(f"{insight}")
    print("\n" + "=" * 95)
    print("🛡️ 风险综述与交易建议")
    print("-" * 95)
    if veto_triggered:
        final_risk = "🔴 红色 (系统性熔断)"
        reason = " + ".join(veto_msgs)
        advice = f"触发熔断: [{reason}]。央行干预或债市危机迫在眉睫。建议做空JGB，持有现金。"
    elif total_score > 6:
        final_risk = "🟠 橙色 (高压警戒)"
        advice = "输入性通胀压力巨大，且YCC调整预期强烈。回避长债和内需股，关注出口/银行。"
    elif total_score > 3:
        final_risk = "🟡 黄色 (震荡)"
        advice = "处于通胀与复苏的博弈期。关注半导体(T)和商社股，警惕汇率波动。"
    else:
        final_risk = "🟢 绿色 (安全/复苏)"
        advice = "宏观环境宽松，日元汇率稳定。适合配置日经225指数ETF或半导体龙头。"
    print(f"📊 加权风险分: {total_score:.2f} / 10.0")
    print(f"🏁 最终评级: {final_risk}")
    print(f"💡 交易建议: {advice}")
    print("=" * 95)

    # 生成HTML报告
    html_file = generate_html_report_japan(indicators, total_score, final_risk, advice, veto_triggered, veto_msgs)

    return {
        "total_score": total_score,
        "final_risk": final_risk,
        "advice": advice,
        "veto_triggered": veto_triggered,
        "veto_msgs": veto_msgs,
        "html_file": html_file
    }


if __name__ == "__main__":
    try:
        data = get_japan_indicators()
        result = generate_report(data)
        print(f"\n📄 HTML报告已保存至: {result['html_file']}")
    except Exception as e:
        print(f"Critical Error: {e}")