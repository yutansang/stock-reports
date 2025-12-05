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
        """智能对齐：处理比率分析中的日期错位"""
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
        # 1. 计算年线
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
                          inverse=False, is_ratio=False,
                          ratio_num=None, ratio_den=None,
                          fallback_ticker=None, external_series=None):  # 支持直接传入Series
        try:
            series = None
            display_ticker = ticker

            # --- 模式A: 外部序列 ---
            if external_series is not None:
                series = external_series
                display_ticker = "Composite"
            # --- 模式B: 比率分析 ---
            elif is_ratio:
                s_num = self.fetch_data_safe(ratio_num)
                s_den = self.fetch_data_safe(ratio_den)
                if s_num.empty or s_den.empty:
                    raise ValueError("比率数据源缺失")

                s_num_aligned, s_den_aligned = self.align_time_series(s_num, s_den)
                if len(s_num_aligned) < self.min_data_points:
                    raise ValueError("对齐后长度不足")

                series = s_num_aligned / s_den_aligned
                display_ticker = f"{ratio_num}/{ratio_den}"

            # --- 模式C: 单资产 ---
            else:
                series = self.fetch_data_safe(ticker)
                # 备用处理逻辑
                if (series.empty or len(series) < self.min_data_points) and fallback_ticker:
                    print(f"⚠️ [{name}] 主代码 {ticker} 无效，切换备用: {fallback_ticker}")
                    series = self.fetch_data_safe(fallback_ticker)
                    display_ticker = fallback_ticker

                if series.empty:
                    raise ValueError("数据源完全失效")
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


def get_us_indicators():
    print("🔍 正在扫描美国宏观经济 ESPT 指标 (Final Robust Ver)...")
    indicators = {"E": [], "S": [], "P": [], "T": []}
    # === E: 预期 (Expectation) ===
    # 1. 衰退交易 (XLY/XLP)
    indicators["E"].append(analyzer.fetch_and_analyze(
        name="衰退交易 (XLY/XLP)", ratio_num="XLY", ratio_den="XLP", is_ratio=True,
        rationale="逻辑: 比率崩塌 = 资金押注硬着陆，防御板块受宠。",
        inverse=True  # 比率跌 -> 风险高
    ))

    # 2. 恐慌指数 (VIX)
    indicators["E"].append(analyzer.fetch_and_analyze(
        name="恐慌指数 (VIX)", ticker="^VIX", fallback_ticker="VIXY",
        rationale="逻辑: 华尔街恐惧指标。飙升通常伴随流动性枯竭。",
        inverse=False  # VIX高 -> 风险高
    ))
    # === S: 结构 (Structure) ===
    # 1. 信用市场 (垃圾债 HYG)
    indicators["S"].append(analyzer.fetch_and_analyze(
        name="信用市场 (HYG)", ticker="HYG", fallback_ticker="JNK",
        rationale="逻辑: 实体经济违约风险。价格崩盘 = 信用冻结。",
        inverse=True  # 价格跌 -> 风险高
    ))
    # 2. 收益率曲线 (10Y-2Y) -- 新增核心指标
    try:
        ten_yr = analyzer.fetch_data_safe("^TNX")
        two_yr = analyzer.fetch_data_safe("^FVX")
        if not ten_yr.empty and not two_yr.empty:
            ten, two = analyzer.align_time_series(ten_yr, two_yr)
            spread = ten - two

            indicators["S"].append(analyzer.fetch_and_analyze(
                name="收益率曲线 (10Y-2Y)", external_series=spread,
                rationale="逻辑: 倒挂加深(负值变大) = 衰退概率激增。",
                inverse=True  # 利差越小(越负) -> 风险越高
            ))
        else:
            raise ValueError("数据不足")
    except:
        indicators["S"].append({"name": "收益率曲线", "level": "gray", "msg": "Error", "rationale": "数据获取失败"})
    # === P: 权力/政策 (Power) ===
    # 1. 美债收益率 (10Y) -- 修复逻辑
    tnx = analyzer.fetch_data_safe("^TNX")
    if not tnx.empty:
        indicators["P"].append(analyzer.fetch_and_analyze(
            name="美债收益率 (10Y)", external_series=tnx,
            rationale="逻辑: 全球定价之锚。急速飙升 = 杀估值。",
            inverse=False  # 收益率高 -> 风险高
        ))
    else:
        # 备用逻辑：使用 TLT (债券价格)，逻辑反转
        print("⚠️ 主代码 ^TNX 无效，切换备用 TLT (逻辑反转)...")
        indicators["P"].append(analyzer.fetch_and_analyze(
            name="美债收益率代理 (TLT)", ticker="TLT",
            rationale="逻辑: (反向) TLT暴跌 = 收益率飙升 = 紧缩恐慌。",
            inverse=True  # TLT跌 -> 收益率涨 -> 风险高
        ))
    # 2. 美元霸权 (DXY)
    indicators["P"].append(analyzer.fetch_and_analyze(
        name="美元流动性 (DXY)", ticker="DX-Y.NYB", fallback_ticker="UUP",
        rationale="逻辑: 强美元收割全球。DXY暴涨 = 全球流动性紧缩。",
        inverse=False  # DXY高 -> 风险高
    ))
    # === T: 技术 (Technology) ===
    # 1. AI引擎 (SMH)
    indicators["T"].append(analyzer.fetch_and_analyze(
        name="AI引擎 (SMH)", ticker="SMH", fallback_ticker="NVDA",
        rationale="逻辑: 美股信仰。AI故事破灭 = 泡沫破裂。",
        inverse=True  # 价格跌 -> 风险高
    ))

    # 2. 流动性探针 (BTC)
    indicators["T"].append(analyzer.fetch_and_analyze(
        name="流动性探针 (BTC)", ticker="BTC-USD",
        rationale="逻辑: 风险偏好最敏感的指标。币圈崩盘领先纳指。",
        inverse=True  # 价格跌 -> 风险高
    ))
    return indicators


def generate_html_report_us(indicators, total_score, final_risk, advice, veto_triggered=False, veto_msgs=None):
    """生成美国ESPT指标的HTML报告（优化排版版）"""

    html_style = """
    <style>
        /* 全局样式重置与基础设置 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body { 
            font-family: "Segoe UI", "Microsoft YaHei", "Arial", sans-serif; 
            background-color: #f0f2f5; 
            color: #333; 
            line-height: 1.6;
        }

        /* 仪表盘容器 */
        .dashboard { 
            max-width: 1200px; 
            margin: 20px auto; 
            background-color: white; 
            border-radius: 16px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.12); 
            overflow: hidden; 
            transition: all 0.3s ease;
        }

        /* 头部样式 */
        .header { 
            background: linear-gradient(135deg, #0d3b66 0%, #1a5f7a 100%); 
            color: white; 
            padding: 35px 40px; 
            text-align: center; 
            position: relative;
        }
        .header::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
        }
        .header h1 { 
            margin: 0 0 15px 0; 
            font-size: 32px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-weight: 600;
        }
        .flag-icon { 
            font-size: 36px; 
            margin-right: 15px; 
        }
        .timestamp { 
            font-size: 15px; 
            opacity: 0.9; 
            margin-top: 8px;
            font-weight: 400;
        }

        /* 风险概览区域 */
        .risk-summary { 
            padding: 30px 40px; 
            background-color: #f8fafc; 
            border-bottom: 1px solid #e2e8f0;
        }
        .risk-title { 
            font-size: 22px; 
            font-weight: 600; 
            margin-bottom: 20px; 
            color: #0d3b66;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .risk-title::before {
            content: "📈";
            font-size: 24px;
        }
        .risk-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
            gap: 20px; 
        }
        .risk-card { 
            background: white; 
            padding: 22px; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.06); 
            border-top: 4px solid #3b82f6;
            transition: transform 0.2s ease;
        }
        .risk-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        }
        .risk-label { 
            font-weight: 500; 
            color: #64748b; 
            margin-bottom: 8px; 
            font-size: 14px; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .risk-value { 
            font-size: 28px; 
            font-weight: 700; 
            margin-top: 4px;
        }

        /* 维度板块样式 */
        .dimension-section { 
            padding: 35px 40px; 
            border-bottom: 1px solid #e2e8f0;
        }
        .dimension-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 25px; 
            flex-wrap: wrap;
            gap: 10px;
        }
        .dimension-title { 
            font-size: 24px; 
            font-weight: 600; 
            color: #1e293b;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .dimension-title::before {
            content: attr(data-icon);
            font-size: 26px;
        }
        .dimension-weight { 
            font-size: 15px; 
            color: #64748b; 
            background-color: #f1f5f9; 
            padding: 8px 18px; 
            border-radius: 20px;
            font-weight: 500;
        }

        /* 指标容器 */
        .indicators-container { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); 
            gap: 20px; 
        }
        @media (max-width: 768px) {
            .indicators-container {
                grid-template-columns: 1fr;
            }
        }
        .indicator-box { 
            padding: 22px; 
            border-radius: 12px; 
            background-color: white; 
            border-left: 5px solid #94a3b8; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.04); 
            transition: all 0.2s ease;
        }
        .indicator-box:hover { 
            transform: translateY(-3px); 
            box-shadow: 0 6px 18px rgba(0,0,0,0.08); 
        }
        .indicator-top { 
            display: flex; 
            justify-content: space-between; 
            align-items: flex-start; 
            margin-bottom: 15px; 
            flex-wrap: wrap;
            gap: 10px;
        }
        .indicator-name { 
            font-weight: 600; 
            font-size: 17px; 
            color: #1e293b;
        }
        .indicator-ticker { 
            font-size: 13px; 
            color: #64748b; 
            background-color: #f1f5f9; 
            padding: 4px 12px; 
            border-radius: 12px;
            font-weight: 500;
        }

        /* 指标数据统计 */
        .indicator-stats { 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 12px; 
            margin-bottom: 15px; 
        }
        .stat-item { 
            display: flex; 
            justify-content: space-between; 
            padding: 10px 0; 
            border-bottom: 1px solid #f1f5f9;
        }
        .stat-label { 
            color: #64748b; 
            font-weight: 500; 
            font-size: 14px;
        }
        .stat-value { 
            font-weight: 600; 
            font-size: 15px;
        }

        /* 原理解读 */
        .indicator-rationale { 
            font-size: 14px; 
            color: #475569; 
            line-height: 1.7; 
            padding: 15px; 
            background-color: #f8fafc; 
            border-radius: 8px; 
            margin-top: 10px; 
            border-left: 3px solid #cbd5e1;
        }

        /* 状态等级颜色 */
        .level-red { color: #dc2626; font-weight: 600; }
        .level-orange { color: #ea580c; font-weight: 600; }
        .level-green { color: #16a34a; font-weight: 600; }
        .level-yellow { color: #ca8a04; font-weight: 600; }
        .level-gray { color: #94a3b8; font-weight: 600; }

        /* 熔断预警 */
        .veto-alert { 
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
            border-left: 6px solid #dc2626; 
            padding: 25px; 
            margin: 25px 40px; 
            border-radius: 12px;
        }
        .veto-title { 
            color: #dc2626; 
            font-weight: 600; 
            font-size: 18px; 
            margin-bottom: 10px; 
            display: flex; 
            align-items: center; 
        }
        .veto-icon { 
            margin-right: 10px; 
            font-size: 20px; 
        }

        /* 页脚样式 */
        .footer { 
            padding: 30px 40px; 
            text-align: center; 
            color: #64748b; 
            font-size: 14px; 
            border-top: 1px solid #e2e8f0; 
            background-color: #f8fafc;
        }
        .color-legend { 
            display: flex; 
            justify-content: center; 
            gap: 20px; 
            margin-top: 20px; 
            flex-wrap: wrap;
        }
        .legend-item { 
            display: flex; 
            align-items: center; 
            gap: 8px;
            font-size: 13px;
        }
        .legend-color { 
            width: 18px; 
            height: 18px; 
            border-radius: 4px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }

        /* 响应式适配 */
        @media (max-width: 992px) {
            .dashboard {
                margin: 10px;
                border-radius: 12px;
            }
            .header {
                padding: 25px 20px;
            }
            .header h1 {
                font-size: 26px;
            }
            .risk-summary, .dimension-section {
                padding: 25px 20px;
            }
            .veto-alert {
                margin: 20px 20px;
                padding: 20px;
            }
        }

        @media (max-width: 576px) {
            .header h1 {
                font-size: 22px;
                flex-direction: column;
                gap: 8px;
            }
            .flag-icon {
                margin-right: 0;
            }
            .risk-value {
                font-size: 24px;
            }
            .dimension-title {
                font-size: 20px;
            }
            .indicator-stats {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """

    # 颜色映射
    color_map = {
        "red": "#dc2626", "orange": "#ea580c",
        "green": "#16a34a", "yellow": "#ca8a04", "gray": "#94a3b8"
    }

    icon_map = {
        "red": "🔴", "orange": "🟠", "yellow": "🟡",
        "green": "🟢", "gray": "⚪"
    }

    dimension_titles = {"E": "预期 (Expectation)", "S": "结构 (Structure)",
                        "P": "权力/政策 (Power)", "T": "技术 (Technology)"}
    dimension_weights = {"E": 0.20, "S": 0.30, "P": 0.30, "T": 0.20}
    dimension_icons = {"E": "📊", "S": "🏗️", "P": "🏛️", "T": "💻"}

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
            <h1><span class='flag-icon'>🇺🇸</span>美国宏观经济 ESPT 仪表盘</h1>
            <p class='timestamp'>华尔街深度分析 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class='risk-summary'>
            <div class='risk-title'>宏观风险概览</div>
            <div class='risk-grid'>
                <div class='risk-card'>
                    <div class='risk-label'>综合风险评分</div>
                    <div class='risk-value' style="color: {color_map.get(risk_color, '#0d3b66')}">
                        {total_score:.2f} / 10.0
                    </div>
                </div>
                <div class='risk-card'>
                    <div class='risk-label'>系统风险评级</div>
                    <div class='risk-value'>{final_risk}</div>
                </div>
                <div class='risk-card'>
                    <div class='risk-label'>交易策略建议</div>
                    <div class='risk-value' style="font-size: 18px; color: #334155; line-height: 1.5;">{advice[:50]}...</div>
                </div>
            </div>
        </div>
    """

    if veto_triggered and veto_msgs:
        html_body += f"""
        <div class='veto-alert'>
            <div class='veto-title'><span class='veto-icon'>⚠️</span> 系统性风险熔断触发</div>
            <div>检测到多重风险共振: <strong>{' + '.join(veto_msgs)}</strong></div>
        </div>
        """

    # 各维度指标展示
    for dim, items in indicators.items():
        dim_title = dimension_titles.get(dim, dim)
        dim_weight = dimension_weights.get(dim, 0.25)
        dim_icon = dimension_icons.get(dim, "📌")

        html_body += f"""
        <div class='dimension-section'>
            <div class='dimension-header'>
                <div class='dimension-title' data-icon="{dim_icon}">{dim_title}</div>
                <div class='dimension-weight'>权重: {dim_weight*100:.0f}%</div>
            </div>
            
            <div class='indicators-container'>
        """

        for item in items:
            level = item.get('level', 'gray')
            icon = icon_map.get(level, '⚪')
            border_color = color_map.get(level, '#94a3b8')

            html_body += f"""
            <div class='indicator-box' style="border-left-color: {border_color};">
                <div class='indicator-top'>
                    <div class='indicator-name'>{icon} {item.get('name', 'N/A')}</div>
                    <div class='indicator-ticker'>{item.get('ticker', 'N/A')}</div>
                </div>
                
                <div class='indicator-stats'>
                    <div class='stat-item'>
                        <span class='stat-label'>当前值:</span>
                        <span class='stat-value'>{item.get('current', '-')}</span>
                    </div>
                    <div class='stat-item'>
                        <span class='stat-label'>乖离率:</span>
                        <span class='stat-value'>{item.get('bias', '-')}</span>
                    </div>
                    <div class='stat-item'>
                        <span class='stat-label'>Z-Score:</span>
                        <span class='stat-value level-{level}'>{item.get('z', 0):+.2f}σ</span>
                    </div>
                    <div class='stat-item'>
                        <span class='stat-label'>状态:</span>
                        <span class='stat-value level-{level}'>{item.get('msg', '-')}</span>
                    </div>
                </div>
                
                <div class='indicator-rationale'>
                    <strong>原理解读:</strong> {item.get('rationale', '无原理解读')}
                </div>
            </div>
            """

        html_body += """
            </div>
        </div>
        """

    # 颜色图例
    html_body += """
    <div class='footer'>
        <div class='color-legend'>
            <div class='legend-item'>
                <div class='legend-color' style="background-color: #dc2626;"></div>
                <span>红色: 极度异常 (风险极高)</span>
            </div>
            <div class='legend-item'>
                <div class='legend-color' style="background-color: #ea580c;"></div>
                <span>橙色: 显著偏离 (风险高)</span>
            </div>
            <div class='legend-item'>
                <div class='legend-color' style="background-color: #ca8a04;"></div>
                <span>黄色: 处于均值 (风险中等)</span>
            </div>
            <div class='legend-item'>
                <div class='legend-color' style="background-color: #16a34a;"></div>
                <span>绿色: 低位安全 (风险低)</span>
            </div>
        </div>
        <p style="margin-top: 20px;">ESPT分析框架 | 美国宏观经济仪表盘 | 基于乖离率Z-Score算法</p>
        <p style="font-size: 13px; color: #94a3b8; margin-top: 5px;">数据来源: Yahoo Finance | 免责声明: 本报告仅供参考，不构成投资建议</p>
    </div>
    </div>
    </body>
    """

    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>US Macroeconomic ESPT Dashboard</title>
        {html_style}
    </head>
    {html_body}
    </html>
    """

    filename = "usa_econ_report.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"\n✅ 生成HTML报告: {filename}")
    return filename


def generate_report(indicators):
    print("\n" + "=" * 95)
    print("🇺🇸 美国宏观经济 ESPT 仪表盘 (Final Production Ver)")
    print("=" * 95)
    weights = {"E": 0.20, "S": 0.30, "P": 0.30, "T": 0.20}
    score_map = {"red": 10, "orange": 6, "yellow": 3, "green": 0, "gray": 5}

    total_score = 0
    veto_msgs = []
    insights = []

    st = {}  # 状态追踪
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

            # 记录状态
            if "VIX" in item['name']:
                st['VIX'] = item['level']
            if "HYG" in item['name']:
                st['Credit'] = item['level']
            if "美债" in item['name']:
                st['Rates'] = item['level']  # 覆盖TNX和TLT
            if "DXY" in item['name']:
                st['Dollar'] = item['level']
        total_score += (dim_score / len(items)) * weights[dim]
    # === 复合熔断逻辑 (Enhanced Veto) ===

    # 1. 红色危机模式 (Red Crisis)
    if st.get('VIX') == 'red' and st.get('Credit') == 'red':
        veto_msgs.append("流动性休克 (VIX spike + Credit freeze)")
    if st.get('Rates') == 'red' and st.get('Dollar') == 'red':
        veto_msgs.append("紧缩风暴 (Rates + Dollar surge)")

    # 2. 橙色早期预警 (Orange Warning)
    if st.get('VIX') in ['red', 'orange'] and st.get('Credit') in ['red', 'orange']:
        if not veto_msgs:  # 避免重复
            veto_msgs.append("早期预警: 流动性压力上升")
    veto_triggered = len(veto_msgs) > 0
    print("\n" + "=" * 95)
    print("🧠 深度逻辑透视 (Deep Dive Analysis)")
    print("-" * 95)
    if not insights:
        print("  华尔街目前处于'金发姑娘'(Goldilocks)状态，主要宏观指标运行平稳。")
    else:
        for insight in insights:
            print(f"{insight}")
    print("\n" + "=" * 95)
    print("🛡️ 风险综述与交易建议")
    print("-" * 95)
    if veto_triggered:
        final_risk = "🔴 红色 (危机模式)"
        reason = " + ".join(veto_msgs)
        advice = f"触发熔断: [{reason}]。这是系统性风险释放信号。清仓股票，买入波动率(VIX)和超短债(SHV)。"
    elif total_score > 6:
        final_risk = "🟠 橙色 (高压)"
        advice = "金融条件收紧。建议缩减科技股敞口，增持现金或防御性板块(XLP)。"
    elif total_score > 3:
        final_risk = "🟡 黄色 (震荡)"
        advice = "多空博弈剧烈。建议哑铃策略：一手AI龙头(SMH)，一手高息债/红利。"
    else:
        final_risk = "🟢 绿色 (Risk-On)"
        advice = "流动性充裕，趋势健康。顺势而为，做多纳指(QQQ)和风险资产。"
    print(f"📊 加权风险分: {total_score:.2f} / 10.0")
    print(f"🏁 最终评级: {final_risk}")
    print(f"💡 交易建议: {advice}")
    print("=" * 95)

    # 生成HTML报告
    html_file = generate_html_report_us(indicators, total_score, final_risk, advice, veto_triggered, veto_msgs)

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
        data = get_us_indicators()
        result = generate_report(data)
        print(f"\n📄 HTML报告已保存至: {result['html_file']}")
    except Exception as e:

        print(f"Critical Error: {e}")
