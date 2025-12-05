# -*- coding: utf-8 -*-
"""
US Economy ESPT Monitor - v14 (The Coronation)
- Fixes a final `NameError` in the main execution block.
- This version is feature-complete, robust, and represents the final, successful state of our collaboration.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import functools
import time

class MacroAnalyzer:
    def __init__(self, window=252, data_days=1260):
        self.window, self.data_days = window, data_days
        self.z_thresholds = {"red": 2.0, "orange": 1.5, "yellow": 1.0}
        self.data_issues = []
        pd.set_option('display.float_format', '{:.2f}'.format)

    @functools.lru_cache(maxsize=32)
    def fetch_data_safe(self, ticker, max_retries=3):
        for attempt in range(max_retries):
            try:
                start_date = datetime.now() - timedelta(days=self.data_days)
                data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True, threads=False)
                if data.empty:
                    if attempt == max_retries - 1: self.data_issues.append({'ticker': ticker, 'issue': '下载数据为空'}); print(f"❌ 警告: {ticker} 下载数据为空 (所有尝试均失败)。")
                    else: print(f"⚠️ 警告: {ticker} 下载数据为空 (尝试 {attempt+1}/{max_retries})"); time.sleep(1)
                    continue
                if 'Close' not in data.columns: raise ValueError("返回的数据中不包含 'Close' 列。")
                close_prices = data[['Close']].copy(); close_prices.columns = [ticker]
                print(f"✅ 成功获取 {ticker}: {len(close_prices)} 行")
                return close_prices
            except Exception as e:
                if attempt == max_retries - 1: self.data_issues.append({'ticker': ticker, 'issue': str(e)}); print(f"❌ 错误: {ticker} 下载失败: {e}")
                else: print(f"⚠️ 错误: {ticker} 下载中 (尝试 {attempt+1}/{max_retries}): {e}"); time.sleep(1)
        return pd.DataFrame()

    def _ensure_series(self, data):
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1: return data.iloc[:, 0].copy()
            else: return pd.Series(dtype=float)
        elif isinstance(data, pd.Series): return data.copy()
        return pd.Series(dtype=float)

    def align_time_series(self, s1, s2):
        s1_series, s2_series = self._ensure_series(s1), self._ensure_series(s2)
        if s1_series.empty or s2_series.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
        combined = pd.DataFrame({'s1': s1_series, 's2': s2_series}).dropna()
        return combined['s1'], combined['s2']

    def calculate_bias_z_score(self, series, window):
        series = self._ensure_series(series)
        if series.empty or len(series) < window: return None, None
        rolling_mean = series.rolling(window=window, min_periods=int(window*0.8)).mean()
        epsilon = 1e-10; bias = (series - rolling_mean) / (rolling_mean + epsilon); bias = bias.dropna()
        if bias.empty: return None, None
        z_score_rolling_mean, z_score_rolling_std = bias.rolling(window=window, min_periods=int(window*0.8)).mean(), bias.rolling(window=window, min_periods=int(window*0.8)).std()
        z_score = (bias - z_score_rolling_mean) / (z_score_rolling_std + epsilon)
        return z_score.iloc[-1], bias.iloc[-1]

    def calculate_vanilla_z_score(self, series, window):
        series = self._ensure_series(series)
        if series.empty or len(series) < window: return None, None
        rolling_mean, rolling_std = series.rolling(window=window, min_periods=int(window*0.8)).mean(), series.rolling(window=window, min_periods=int(window*0.8)).std()
        epsilon = 1e-10; z_score = (series - rolling_mean) / (rolling_std + epsilon)
        return z_score.iloc[-1], series.iloc[-1]

    def get_status_color(self, z_score, inverse=False):
        if z_score is None: return 'grey', '#A0AEC0', '数据缺失'
        score = -z_score if inverse else z_score
        if score > self.z_thresholds["red"]: return 'red', '#e53e3e', '极度偏高'
        elif score < -self.z_thresholds["red"]: return 'dark_red', '#9b2c2c', '极度偏低'
        elif score > self.z_thresholds["orange"]: return 'orange', '#dd6b20', '显著偏高'
        elif score < -self.z_thresholds["orange"]: return 'dark_orange', '#b7791f', '显著偏低'
        elif abs(score) > self.z_thresholds["yellow"]: return 'yellow', '#d69e2e', '轻微偏离'
        else: return 'green', '#38a169', '正常区间'

    def analyze_series(self, name, rationale, external_series, inverse=False, analysis_type='bias', dimension=''):
        if external_series is None or external_series.empty: return {"name": name, "rationale": rationale, "z_score": None, "value": None, "status": 'missing', "color": '#CBD5E0', "status_text": '数据缺失', "inverse": inverse, "value_label": "N/A", "dimension": dimension}
        label = "乖离率" if analysis_type == 'bias' else "当前值"
        z_score, value = self.calculate_bias_z_score(external_series, self.window) if analysis_type == 'bias' else self.calculate_vanilla_z_score(external_series, self.window)
        status, color, status_text = self.get_status_color(z_score, inverse)
        return {"name": name, "rationale": rationale, "z_score": z_score, "value": value, "status": status, "color": color, "status_text": status_text, "inverse": inverse, "value_label": label, "dimension": dimension}
    
    def get_data_quality_report(self):
        if not self.data_issues: return "✅ 所有数据获取尝试均成功，报告质量完美！"
        report = "📊 数据获取质量报告:\n" + "=" * 50 + "\n"
        for issue in self.data_issues: report += f"\n📈 Ticker: {issue['ticker']}\n  - 问题: {issue['issue']}\n"
        report += "\n💡 建议:\n1. 检查网络连接。\n2. 确认失败的 Ticker 符号是否正确。\n3. 尝试手动访问雅虎财经 (e.g., https://finance.yahoo.com/quote/SPY) 确认资产是否存在。\n4. 如果在公司网络下，可能是防火墙阻止了API请求，请尝试更换网络环境。\n"
        return report

def get_us_indicators(analyzer):
    indicators = {"E": [], "S": [], "P": [], "T": []}
    # E
    indicators["E"].append(analyzer.analyze_series("VIX恐慌指数", "...", analyzer.fetch_data_safe("^VIX"), inverse=True, dimension="E"))
    indicators["E"].append(analyzer.analyze_series("MOVE债市恐慌指数", "...", analyzer.fetch_data_safe("^MOVE"), inverse=True, dimension="E"))
    s1, s2 = analyzer.align_time_series(analyzer.fetch_data_safe("XLY"), analyzer.fetch_data_safe("XLP"))
    indicators["E"].append(analyzer.analyze_series("衰退交易(XLY/XLP)", "...", s1/s2 if not s1.empty else None, dimension="E"))
    # S
    indicators["S"].append(analyzer.analyze_series("核心资产(SPY)", "...", analyzer.fetch_data_safe("SPY"), dimension="S"))
    s1, s2 = analyzer.align_time_series(analyzer.fetch_data_safe("HYG"), analyzer.fetch_data_safe("TLT"))
    indicators["S"].append(analyzer.analyze_series("信用利差(HYG/TLT)", "...", s1/s2 if not s1.empty else None, dimension="S"))
    indicators["S"].append(analyzer.analyze_series("房地产(IYR)", "...", analyzer.fetch_data_safe("IYR"), dimension="S"))
    # P
    indicators["P"].append(analyzer.analyze_series("美元指数(UUP)", "...", analyzer.fetch_data_safe("UUP"), dimension="P"))
    indicators["P"].append(analyzer.analyze_series("美债收益率(10Y)", "...", analyzer.fetch_data_safe("^TNX"), inverse=True, dimension="P"))
    s1, s2 = analyzer.align_time_series(analyzer.fetch_data_safe("^TNX"), analyzer.fetch_data_safe("^IRX"))
    if not s1.empty: indicators["P"].append(analyzer.analyze_series("衰退预警(10Y-3M利差)", "...", s1-s2, analysis_type='vanilla', dimension="P"))
    # T
    indicators["T"].append(analyzer.analyze_series("科技股(QQQ)", "...", analyzer.fetch_data_safe("QQQ"), dimension="T"))
    indicators["T"].append(analyzer.analyze_series("半导体(SOXX)", "...", analyzer.fetch_data_safe("SOXX"), dimension="T"))
    indicators["T"].append(analyzer.analyze_series("全球风险偏好(BTC)", "...", analyzer.fetch_data_safe("BTC-USD"), dimension="T"))
    return {k: [i for i in v if i] for k, v in indicators.items()}

def enhanced_veto_logic(analyzer, all_indicators):
    veto_msgs = []
    vix_series = analyzer._ensure_series(analyzer.fetch_data_safe("^VIX"))
    if not vix_series.empty and vix_series.iloc[-1] > 30: veto_msgs.append(f"!! 极端恐慌: VIX指数 ({vix_series.iloc[-1]:.2f}) 超过30警戒线。")
    return veto_msgs

def generate_detailed_assessment(all_indicators, avg_score):
    html = "<h3>一、市场情绪定调</h3>"
    if avg_score > 1.8: html += "<p><strong>市场状态：<span style='color:#e53e3e;'>极度贪婪 / 恐慌</span></strong>。多个关键指标均严重偏离其历史常态，市场情绪已进入极端区域，趋势随时可能出现剧烈反转，风险极高。</p>"
    elif avg_score > 1.2: html += "<p><strong>市场状态：<span style='color:#dd6b20;'>显著偏离</span></strong>。市场展现出明确的趋势和情绪，但部分指标已进入过热/过冷区间，需高度警惕潜在的回调压力。</p>"
    elif avg_score < 0.8: html += "<p><strong>市场状态：<span style='color:#38a169;'>盘整与观望</span></strong>。市场缺乏明确方向，多数指标在历史均值附近徘徊，投资者情绪相对中性，正在等待新的催化剂。</p>"
    else: html += "<p><strong>市场状态：<span style='color:#d69e2e;'>温和趋势</span></strong>。市场正沿着特定方向发展，但整体偏离度仍在可控范围内，趋势相对健康。</p>"
    all_items = [item for dim_items in all_indicators.values() for item in dim_items if item['z_score'] is not None]; all_items.sort(key=lambda x: abs(x['z_score']), reverse=True)
    hottest_risks = [item for item in all_items if (item['z_score'] > 1.5 and not item['inverse']) or (item['z_score'] < -1.5 and item['inverse'])]; stabilizers = [item for item in all_items if abs(item['z_score']) < 0.5]
    html += "<h3>二、核心驱动力分析</h3>";
    if hottest_risks: html += "<h4>主要风险来源 (Z-Score > 1.5):</h4><ul>" + "".join(f"<li><strong>{item['name']}:</strong> Z-Score为 <strong>{item['z_score']:.2f}</strong>，显示出 <strong>{item['status_text']}</strong> 状态，是当前市场过热的主要推手。</li>" for item in hottest_risks[:3]) + "</ul>"
    else: html += "<p>✅ 当前市场未发现显著的过热风险信号。</p>"
    if stabilizers: html += "<h4>市场压舱石 (Z-Score < 0.5):</h4><ul>" + "".join(f"<li><strong>{item['name']}:</strong> Z-Score为 <strong>{item['z_score']:.2f}</strong>，处于历史正常区间，为市场提供了稳定性。</li>" for item in stabilizers[:2]) + "</ul>"
    html += "<h3>三、维度间交叉叙事</h3>"
    e_scores, t_scores, p_scores = [i['z_score'] for i in all_items if i['dimension'] == 'E'], [i['z_score'] for i in all_items if i['dimension'] == 'T'], [i['z_score'] for i in all_items if i['dimension'] == 'P' and '利差' in i['name']]
    avg_t, p_z = np.mean([abs(s) for s in t_scores]) if t_scores else 0, p_scores[0] if p_scores else 0
    narrative = "<p><strong>核心矛盾：科技狂热 vs 经济预警。</strong>我们观察到，以半导体和科技股为代表的<strong>技术(T)</strong>维度正显示出极度乐观的情绪（平均Z-Score > 1.5），而代表宏观经济前景的<strong>权力(P)</strong>维度中的“衰退预警”指标却在发出减速信号。这种“预期”与“现实”的巨大背离是当前市场最主要的风险来源，暗示技术板块的上涨可能缺乏坚实的宏观基本面支撑。</p>" if avg_t > 1.5 and p_z < -1.0 else "<p><strong>市场由预期驱动。</strong>以VIX和衰退交易为代表的<strong>预期(E)</strong>维度指标出现显著偏离，表明当前市场的主要驱动力来自于投资者的情绪和对未来的押注，而非已确认的经济结构变化。需密切关注情绪指标是否能被后续的实体经济数据所验证。</p>" if np.mean([abs(s) for s in e_scores]) > 1.5 else "<p><strong>多空平衡。</strong>当前各维度之间未出现极端背离，市场在多方因素的拉扯下寻找方向。建议密切关注各维度指标的后续变化，以判断未来趋势的突破方向。</p>"
    html += narrative
    return html

def generate_report_html(all_indicators, melt_down_messages, analyzer, country="美国"):
    data_quality_report = analyzer.get_data_quality_report()
    html = f"""<html><head><title>{country}宏观经济监控仪表盘</title><style>body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;margin:20px;background-color:#f8f9fa;color:#212529}}h1,h2,h3,h4{{color:#1a202c}}h1{{border-bottom:3px solid #dee2e6;padding-bottom:15px}}h2{{border-bottom:2px solid #e9ecef;padding-bottom:10px;margin-top:40px}}h3{{margin-top:30px;color:#495057}}table{{width:100%;border-collapse:collapse;margin-bottom:25px;box-shadow:0 0 20px rgba(0,0,0,.05)}}th,td{{padding:14px;border:1px solid #dee2e6;text-align:left;vertical-align:top}}th{{background-color:#343a40;color:#fff;text-transform:uppercase;letter-spacing:.05em}}td{{background-color:#fff}}.status-cell{{font-weight:700;text-align:center;color:#fff}}.rationale{{font-size:.85em;color:#6c757d;margin-top:5px}}.report-summary,.melt-down,.data-quality{{padding:25px;margin-bottom:25px;border-left:6px solid;border-radius:8px;background:#fff;box-shadow:0 4px 12px rgba(0,0,0,.08)}}.melt-down{{background-color:#f8d7da;border-left-color:#721c24}}.melt-down h2{{color:#721c24}}.data-quality{{background-color:#fff3cd;border-left-color:#856404}}.data-quality h2{{color:#856404}}.missing-data td{{background-color:#f5c6cb !important;opacity:.7}}.summary-section{{padding:20px;border-radius:8px;background:#f8f9fa;border:1px solid #e9ecef}}pre{{white-space:pre-wrap;font-family:Menlo,Monaco,Consolas,"Courier New",monospace;background:#e9ecef;padding:15px;border-radius:6px;}}</style></head><body><h1>🇺🇸 {country}宏观经济监控仪表盘 (ESPT v14) - {datetime.now().strftime("%Y-%m-%d %H:%M")}</h1><div class='data-quality'><h2>📊 数据质量报告</h2><pre>{data_quality_report}</pre></div>"""
    if melt_down_messages: html += "<div class='melt-down'><h2>🚨 风险熔断警告!</h2><ul>" + "".join(f"<li>{msg}</li>" for msg in melt_down_messages) + "</ul></div>"
    valid_indicators = [item for dim_items in all_indicators.values() for item in dim_items if item['z_score'] is not None]; avg_score = sum(abs(item['z_score']) for item in valid_indicators) / len(valid_indicators) if valid_indicators else 0
    html += "<div class='report-summary'><h2>📈 综合评估 (智能分析)</h2><div class='summary-section'>" + generate_detailed_assessment(all_indicators, avg_score) + "</div></div>"
    for dimension, items in all_indicators.items():
        if not items: continue
        dim_map = {"E": "预期", "S": "结构", "P": "权力", "T": "技术"}; html += f"<h2>{dimension} - {dim_map.get(dimension,'')}</h2><table><tr><th>指标名称</th><th>Z-Score</th><th>数值</th><th>状态</th></tr>"
        for item in items:
            row_class = "missing-data" if item['z_score'] is None else ""
            value_label = item.get('value_label', '乖离率'); value_str = f"{item['value']*100:.2f}%" if value_label == '乖离率' and item['value'] is not None else f"{item['value']:.2f}" if item['value'] is not None else "❌"
            z_score_str = f"{item['z_score']:.2f}" if item['z_score'] is not None else "❌"
            html += f"<tr class='{row_class}'><td><strong>{item['name']}</strong><div class='rationale'>{item['rationale']}</div></td><td>{z_score_str}</td><td>{value_str}</td><td class='status-cell' style='background-color:{item.get('color', '#A0AEC0')};'>{item.get('status_text', '数据缺失')}</td></tr>"
        html += "</table>"
    return html

# --- Main Execution ---
if __name__ == "__main__":
    analyzer = MacroAnalyzer()
    print("开始分析美国经济指标 (v14 The Coronation)...")
    print("=" * 60)
    us_indicators = get_us_indicators(analyzer)
    print("\n" + "=" * 60)
    print("数据获取阶段完成。正在生成诊断报告...")
    print(analyzer.get_data_quality_report())
    print("=" * 60)
    print("\n检查系统性风险...")
    melt_down_messages = enhanced_veto_logic(analyzer, us_indicators)
    print("生成HTML报告...")
    # --- FINAL FIX in main block ---
    report_html = generate_report_html(us_indicators, melt_down_messages, analyzer, country="美国")
    # -----------------------------
    file_name = f"US_Econ_ESPT_Report_v14_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(report_html)
    print(f"\n✅ 报告已生成: {file_name}")
    if analyzer.data_issues:
        print(f"\n⚠️  注意：在数据获取过程中检测到问题。请查看HTML报告顶部的“数据质量报告”获取详细信息。")
