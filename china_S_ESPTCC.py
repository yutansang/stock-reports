import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# =============================================================================
# 1. 配置模块 - "战略司令部" (融合版)
# =============================================================================

# --- 宏观"冠军组合" ---
CHAMPION_TICKERS = {
    'P': {"券商ETF": "512000.SS", "金融地产ETF": "510650.SS"},
    'S': {"银行ETF": "515290.SS", "主要消费ETF": "159928.SZ", "资源ETF": "510410.SS"}, # 银行ETF更新为515290.SS
    'E': {'spear': {"创业板ETF": "159915.SZ", "半导体ETF": "512480.SS"}, 'shield': {"红利ETF": "510850.SS", "医药ETF": "159929.SZ"}},
    'T': {"沪深300ETF": "510300.SS", "中证500ETF": "510500.SS", "上证50ETF": "510050.SS"}
}

# --- 龙头ETF产业链深度下钻配置 ---
DEEP_DIVE_CONFIG = {
    "510410.SS": {
        "chain_name": "资源与周期产业链",
        "baskets": {
            "上游 (纯资源)": {"有色金属ETF": "512400.SS", "煤炭ETF": "515220.SS"},
            "中游 (材料加工)": {"化工ETF": "516020.SS", "钢铁ETF": "515210.SS"},
            "下游 (工业应用)": {"机械ETF": "516960.SS", "基建ETF": "516950.SS"}
        }
    },
    "159915.SZ": {
        "chain_name": "创业板核心成分",
        "baskets": {
            "引擎 (新能源)": {"新能源ETF": "515700.SS"},
            "科技 (硬核)": {"半导体ETF": "512480.SS"},
            "健康 (生物)": {"生物医药ETF": "159929.SZ"}
        }
    },
    "512480.SS": {
        "chain_name": "硬核科技产业链",
        "baskets": {
            "上游 (设备材料)": {"半导体ETF": "512480.SS", "芯片ETF": "159995.SZ"},
            "中游 (平台软件)": {"计算机ETF": "512720.SS", "软件ETF": "515230.SH"}, # .SH
            "下游 (终端应用)": {"人工智能ETF": "159819.SZ", "通信ETF": "515880.SS"}
        }
    },
    "159928.SZ": {
        "chain_name": "大消费产业链",
        "baskets": {
            "上游 (原料)": {"食品饮料ETF": "515170.SS"},
            "中游 (品牌制造)": {"家电ETF": "159996.SZ", "白酒ETF": "512690.SS"},
            "下游 (服务零售)": {"互联网ETF": "517200.SS", "传媒ETF": "512980.SS"}
        }
    },
    "159929.SZ": {
        "chain_name": "医药子行业",
        "baskets": {
            "创新药 (高风险)": {"创新药ETF": "159992.SZ"},
            "器械 (稳健增长)": {"医疗器械ETF": "159883.SZ"},
            "中药 (传统价值)": {"中药ETF": "515920.SS"}
        }
    },
     "512000.SS": {
        "chain_name": "大金融内部轮动",
        "baskets": {
            "进攻 (券商)": {"券商ETF": "512000.SS", "证券ETF": "512880.SS"},
            "稳健 (金融地产)": {"金融地产ETF": "510650.SS"},
            "基石 (银行)": {"银行ETF": "515290.SS"} # 银行ETF更新为515290.SS
        }
    },
    "510850.SS": {
        "chain_name": "价值风格光谱",
        "baskets": {
            "高股息 (纯粹)": {"红利ETF": "510850.SS"},
            "低估值 (广义)": {"价值ETF": "510030.SS"},
            "稳定盈利 (质量)": {"质量ETF": "159935.SZ"}
        }
    },
    "510300.SS": {
        "chain_name": "市场风格因子",
        "baskets": {
            "大盘价值": {"上证50ETF": "510050.SS", "价值ETF": "510030.SS"},
            "大盘成长": {"沪深300成长ETF": "510330.SS"},
            "中小盘": {"中证500ETF": "510500.SS", "中证1000ETF": "159845.SZ"}
        }
    }
}
# --- 为其他龙头引用现有方案 ---
DEEP_DIVE_CONFIG["515290.SS"] = {"_ref": "512000.SS"} # 银行ETF更新
DEEP_DIVE_CONFIG["510650.SS"] = {"_ref": "512000.SS"}
DEEP_DIVE_CONFIG["510500.SS"] = {"_ref": "510300.SS"}
DEEP_DIVE_CONFIG["510050.SS"] = {"_ref": "510300.SS"}

# --- 报告文本与周期参数 ---
descriptions_map = {
    'P': '<b>权力 (Power)</b> - 政策共识', 'S': '<b>结构 (Structure)</b> - 经济共识',
    'E': '<b>预期 (Expectations)</b> - 风险偏好', 'T': '<b>技术 (Technology)</b> - 市场趋势共识'
}
z_score_period = 252
start_date = (datetime.now() - timedelta(days=z_score_period * 2.0)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# =============================================================================
# 2. 核心功能与辅助函数
# =============================================================================

def interpret_signals(z):
    """
    根据宏观共识Z-Score解读当前市场信号
    """
    e_score, s_score, p_score, t_score = z.get('E', 0), z.get('S', 0), z.get('P', 0), z.get('T', 0)

    if p_score > 0.8 and e_score > 0.8 and t_score > 0.8:
        return "趋势共振 (Risk-On)", "<p><b>解读:</b> 政策、风险偏好与市场趋势形成向上合力，市场处于明确的“风险开启”模式。</p><p><b>策略:</b> 积极寻找领涨板块，顺势而为。</p>"
    if e_score < -0.8 and t_score < -0.8 and p_score < -0.5:
        return "趋势共振 (Risk-Off)", "<p><b>解读:</b> 风险偏好、市场趋势与政策预期均偏向悲观，市场处于明确的“风险关闭”模式。</p><p><b>策略:</b> 降低仓位，转向防御性板块或持币观望。</p>"

    return "震荡市/结构分化", "<p><b>解读:</b> 市场缺乏明确方向，各项核心力量相互拉扯。<b>请重点关注下方'个体ETF强度排行榜'，寻找结构性机会。</b></p><p><b>策略:</b> 多看少动，或跟随排行榜顶端的强势ETF进行短线交易。</p>"

def get_bar_color(value):
    """
    根据Z-Score值返回对应的颜色
    """
    if value is None or np.isnan(value): return "#888"
    if value > 1.5: return "#d62728"
    if value > 0.8: return "#ff7f0e"
    if value < -1.5: return "#2ca02c"
    if value < -0.8: return "#1f77b4"
    return "#9467bd"

def run_deep_dive_analysis(leader_code, leader_name):
    """
    对领涨龙头进行产业链深度分析
    """
    config = DEEP_DIVE_CONFIG.get(leader_code)
    if config and "_ref" in config:  config = DEEP_DIVE_CONFIG.get(config["_ref"])
    if not config:
        print(f"龙头 '{leader_name}' 未配置深度分析，跳过。")
        return None

    print(f"\n--- 🚀 第二级火箭启动：对'{config['chain_name']}'进行深度分析 ---")
    deep_dive_codes = []
    for etfs in config["baskets"].values(): deep_dive_codes.extend(etfs.values())
    unique_deep_codes = sorted(list(set(deep_dive_codes)))

    print(f"下载 {len(unique_deep_codes)} 个产业链ETF数据...")
    try:
        deep_data = yf.download(unique_deep_codes, start=start_date, end=end_date, auto_adjust=True, group_by='ticker', progress=False)
        failed_downloads = [code for code in unique_deep_codes if code not in deep_data.columns or deep_data[code].isnull().all().all()]
        if failed_downloads:
            print(f"\033[91m产业链ETF下载失败: {', '.join(failed_downloads)}。深度分析中止。\033[0m")
            return None

        deep_close_data = pd.DataFrame({code: deep_data[code]['Close'] for code in unique_deep_codes if not deep_data[code].empty})
        deep_close_data.ffill(inplace=True)
        deep_close_data.dropna(axis=1, how='all', inplace=True) # 删除所有值为NaN的列
        deep_close_data.dropna(inplace=True) # 删除包含NaN的行


        if len(deep_close_data) < z_score_period * 0.8:
            print("产业链ETF数据共同历史过短，无法进行深度分析。")
            return None

        deep_z_scores = []
        for stage, etfs in config["baskets"].items():
            for name, code in etfs.items():
                if code in deep_close_data.columns:
                    series = deep_close_data[code]
                    mean = series.rolling(window=z_score_period, min_periods=int(z_score_period*0.8)).mean()
                    std = series.rolling(window=z_score_period, min_periods=int(z_score_period*0.8)).std()
                    if pd.notna(std.iloc[-1]) and std.iloc[-1] > 0:
                        z = (series.iloc[-1] - mean.iloc[-1]) / std.iloc[-1]
                    else:
                        z = 0
                    deep_z_scores.append({"name": name, "code": code, "stage": stage, "z_score": z})

        deep_z_scores.sort(key=lambda x: x['z_score'], reverse=True)
        print("产业链深度分析完成！")
        return {"chain_name": config["chain_name"], "data": deep_z_scores}
    except Exception as e:
        print(f"深度分析过程中发生错误: {e}")
        return None

# =============================================================================
# 3. 主逻辑执行区
# =============================================================================

# --- 第一级：宏观广度扫描 ---
print("--- 🛰️ 第一级火箭：进行宏观广度扫描 ---")
all_champion_codes = []
for factor, etfs in CHAMPION_TICKERS.items():
    if factor == 'E':
        all_champion_codes.extend(etfs['spear'].values())
        all_champion_codes.extend(etfs['shield'].values())
    else:
        all_champion_codes.extend(etfs.values())
unique_champion_codes = sorted(list(set(all_champion_codes)))

print(f"下载 {len(unique_champion_codes)} 个'冠军'ETF的数据...")
try:
    champion_data = yf.download(unique_champion_codes, start=start_date, end=end_date, auto_adjust=True, group_by='ticker', progress=False)
    close_data = pd.DataFrame({code: champion_data[code]['Close'] for code in unique_champion_codes if not champion_data[code].empty})
    close_data.ffill(inplace=True);
    close_data.dropna(axis=1, how='all', inplace=True) # 删除全为NaN的列
    close_data.dropna(inplace=True) # 删除包含NaN的行
    if len(close_data) < z_score_period * 0.8:
        print(f"\n\033[93m警告：宏观ETF有效数据共同交易日 ({len(close_data)}) 不足({int(z_score_period * 0.8)}天)。分析终止。\033[0m")
        sys.exit()
    print("宏观数据下载成功！")
except Exception as e:
    print(f"\n\033[91m宏观分析数据下载错误: {e}\033[0m")
    sys.exit()

print("计算个体及共识Z-Score...")
individual_z_scores = {}
for code in close_data.columns:
    series = close_data[code]
    mean = series.rolling(window=z_score_period, min_periods=int(z_score_period*0.8)).mean()
    std = series.rolling(window=z_score_period, min_periods=int(z_score_period*0.8)).std()
    
    # --- 这是修正后的关键代码 ---
    if pd.notna(std.iloc[-1]) and std.iloc[-1] > 0:
        individual_z_scores[code] = (series.iloc[-1] - mean.iloc[-1]) / std.iloc[-1]
    else:
        individual_z_scores[code] = 0
    # --- 修正结束 ---

consensus_z_scores = {}
for factor in ['P', 'S', 'T']:
    consensus_z_scores[factor] = np.mean([individual_z_scores.get(c, 0) for c in CHAMPION_TICKERS[factor].values()])
spear_avg_z = np.mean([individual_z_scores.get(c, 0) for c in CHAMPION_TICKERS['E']['spear'].values()])
shield_avg_z = np.mean([individual_z_scores.get(c, 0) for c in CHAMPION_TICKERS['E']['shield'].values()])
consensus_z_scores['E'] = spear_avg_z - shield_avg_z
consensus_z_scores['Total'] = np.mean(list(consensus_z_scores.values()))
print("Z-Score计算完成！")


# --- 准备ETF排行榜数据 ---
etf_ranking_data = []
code_to_details = {}
for factor, etfs in CHAMPION_TICKERS.items():
    if factor == 'E':
        for name, code in etfs['spear'].items(): code_to_details[code] = {"name": name, "factor": "E (矛)"}
        for name, code in etfs['shield'].items(): code_to_details[code] = {"name": name, "factor": "E (盾)"}
    else:
        for name, code in etfs.items(): code_to_details[code] = {"name": name, "factor": factor}

for code, z in individual_z_scores.items():
    details = code_to_details.get(code)
    if details: etf_ranking_data.append({"name": details["name"], "code": code, "factor": details["factor"], "z_score": z})
etf_ranking_data.sort(key=lambda x: x['z_score'], reverse=True)

# --- 进行交易信号解读 ---
print("正在生成交易信号解读...")
signal_type, interpretation_html = interpret_signals(consensus_z_scores)
print(f"识别到的信号类型: {signal_type}")

# --- 第二级：识别龙头并进行深度分析 ---
leader_etf = etf_ranking_data[0] if etf_ranking_data else None
deep_dive_result = None
if leader_etf:
    leader_code_to_analyze = leader_etf["code"]
    if leader_etf["code"] == "159928.SZ" and leader_etf["factor"] != "S":
         # 159928.SZ 在 S 因子中代表消费，但在其他情况（如 T 因子）下可能需要被视为金融板块进行分析
         # 这里我们假设如果它不是作为“消费”领涨，就按金融分析
         leader_code_to_analyze = "515290.SS" # 指向新的银行ETF代码

    deep_dive_result = run_deep_dive_analysis(leader_code_to_analyze, leader_etf["name"])

# =============================================================================
# 4. HTML报告生成模块 (融合版)
# =============================================================================
print("\n--- 正在生成最终HTML报告 ---")

html_content = f"""
<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>A股轮动仪表盘 (融合战略版)</title>
<style>
    body{{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f0f2f5; color: #333; margin: 20px;}}
    .container{{max-width: 900px; margin: auto; background-color: #fff; padding: 20px 40px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}}
    h1, h2, h3 {{color: #1a2c5b; border-bottom: 2px solid #eef2f7; padding-bottom: 10px;}}
    .espt-card {{border: 1px solid #ddd; border-radius: 5px; margin-bottom: 15px; padding:15px;}}
    .bar-container{{width:100%; background-color:#f1f1f1; border-radius:5px; overflow: hidden;}}
    .bar{{height:24px; line-height:24px; color:white; text-align:right; padding-right:10px; font-weight:bold; white-space:nowrap;}}
    .rank-table table {{width: 100%; border-collapse: collapse;}}
    .rank-table th, .rank-table td {{padding: 12px 15px; text-align: left; border-bottom: 1px solid #eef2f7;}}
    .rank-table th {{background-color: #f8f9fa;}}
    .signal-box{{background-color:#eef2f7; border-left: 5px solid #0056b3; padding:20px; margin-top:30px; margin-bottom: 30px; border-radius:8px;}}
</style>
</head><body><div class="container">
<h1>A股轮动仪表盘 (融合战略版)</h1><p>报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>第一级：宏观共识扫描</h2>
<h3>宏观共识Z-Score</h3>
"""

for k in ['P', 'S', 'E', 'T']:
    v = consensus_z_scores.get(k, 0)
    color = get_bar_color(v)
    width_percentage = min(abs(v) / 2.5, 1) * 100
    html_content += f"""<div class="espt-card"><h4>{descriptions_map[k]}</h4>
    <div class="bar-container"><div class="bar" style="width:{width_percentage}%; background-color:{color};">{v:.2f}</div></div></div>"""

html_content += f"""<div class="signal-box"><h3>交易信号解读: <span>{signal_type}</span></h3>{interpretation_html}</div>"""

html_content += """
<h3>宏观个体ETF强度排行榜 (Z-Score)</h3>
<p>此榜单将所有“冠军组合”ETF按当前强度(Z-Score)降序排列，帮您快速识别领涨龙头和落后板块。</p>
<div class="rank-table"><table>
<thead><tr><th>排名</th><th>ETF名称</th><th>所属指标</th><th>Z-Score</th></tr></thead>
<tbody>
"""
for i, item in enumerate(etf_ranking_data):
    z = item['z_score']
    color = get_bar_color(z)
    width_percentage = min(abs(z) / 3, 1) * 100
    html_content += f"""
    <tr>
        <td>{i+1}</td>
        <td>{item['name']} ({item['code']})</td>
        <td>{item['factor']}</td>
        <td>
            <div class="bar-container">
                <div class="bar" style="width:{width_percentage}%; background-color:{color};">{z:.2f}</div>
            </div>
        </td>
    </tr>
    """
html_content += "</tbody></table></div>"

if deep_dive_result:
    html_content += f"<h2 style='margin-top: 40px;'>第二级：龙头产业链深度分析 ({deep_dive_result['chain_name']})</h2>"
    html_content += "<p>基于宏观排行榜的领涨龙头，对其所在的产业链/赛道进行强度拆解。</p>"
    html_content += "<div class='rank-table'><table><thead><tr><th>排名</th><th>ETF名称</th><th>产业链环节/核心成分</th><th>Z-Score</th></tr></thead><tbody>"
    for i, item in enumerate(deep_dive_result['data']):
        z = item['z_score']
        color = get_bar_color(z)
        width_percentage = min(abs(z) / 3.0, 1) * 100
        html_content += f"""
        <tr>
            <td>{i+1}</td>
            <td>{item['name']} ({item['code']})</td>
            <td>{item['stage']}</td>
            <td>
                <div class="bar-container">
                    <div class="bar" style="width:{width_percentage}%; background-color:{color};">{z:.2f}</div>
                </div>
            </td>
        </tr>
        """
    html_content += "</tbody></table></div>"

html_content += "</div></body></html>"

filename = "CC.html"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n\033[92m报告生成成功！文件已保存为: {filename}\033[0m")
