"""
미국 다이어트 유산균 트렌드 분석
- CSV 파일: 같은 폴더(Global_trend)
- 결과 저장: output 하위 폴더
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 한글 폰트 설정
# ============================================================================

import matplotlib.font_manager as fm

font_list = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic"]
available_font = None

for font_name in font_list:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        available_font = font_name
        break

if available_font:
    plt.rcParams["font.family"] = available_font
    print(f"✓ 한글 폰트 설정: {available_font}")
else:
    plt.rcParams["font.family"] = "sans-serif"
    print("⚠ 한글 폰트를 찾을 수 없습니다.")

plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

print("=" * 80)
print("미국 다이어트 유산균 트렌드 분석")
print("=" * 80)

# ============================================================================
# 폴더 및 경로 설정
# ============================================================================

# 현재 스크립트가 있는 폴더 (Global_trend)
script_dir = os.path.dirname(os.path.abspath(__file__))

# output 폴더 경로 (Global_trend/output)
output_folder = os.path.join(script_dir, "output")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"\n✓ output 폴더 생성: {output_folder}")
else:
    print(f"\n✓ output 폴더 확인: {output_folder}")

# ============================================================================
# CSV 파일 읽기 (같은 폴더)
# ============================================================================

# Global_trend 폴더 내의 CSV 파일 (같은 위치)
csv_files = {
    "probiotics weight loss": os.path.join(script_dir, "probiotics_weight_loss.csv"),
    "diet probiotics": os.path.join(script_dir, "diet_probiotics.csv"),
    "gut health weight": os.path.join(script_dir, "gut_health_weight.csv"),
    "microbiome diet": os.path.join(script_dir, "microbiome_diet.csv"),
}

print(f"\n[분석할 파일 확인]")
print(f"현재 폴더: {script_dir}")
print()

available_files = {}
for keyword, filepath in csv_files.items():
    if os.path.exists(filepath):
        available_files[keyword] = filepath
        print(f"  ✓ {keyword}: {os.path.basename(filepath)}")
    else:
        print(f"  ✗ {keyword}: {os.path.basename(filepath)} (파일 없음)")

if not available_files:
    print("\n" + "=" * 80)
    print("⚠ CSV 파일이 없습니다!")
    print("=" * 80)
    print(f"\n현재 폴더: {script_dir}")
    print("\n필요한 파일:")
    for keyword, filepath in csv_files.items():
        print(f"  • {os.path.basename(filepath)}")
    print("\n위 파일들을 global_trend.py와 같은 폴더에 저장하세요.")
    print("=" * 80)
    exit()

print(f"\n✓ 총 {len(available_files)}개 파일 발견")

# ============================================================================
# 데이터 읽기
# ============================================================================

print("\n" + "=" * 80)
print("데이터 읽기")
print("=" * 80)

all_data = {}
combined_df = pd.DataFrame()

for keyword, filepath in available_files.items():
    try:
        df = pd.read_csv(filepath, skiprows=2)

        date_col = df.columns[0]
        value_col = df.columns[1]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        df[value_col] = df[value_col].replace("<1", "0.5")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        df_clean = pd.DataFrame({"Date": df[date_col], keyword: df[value_col]})
        df_clean.set_index("Date", inplace=True)

        all_data[keyword] = df_clean

        if combined_df.empty:
            combined_df = df_clean.copy()
        else:
            combined_df = combined_df.join(df_clean, how="outer")

        print(f"✓ {keyword}: {len(df_clean)}개 데이터")

    except Exception as e:
        print(f"✗ {keyword} 읽기 실패: {e}")

if combined_df.empty:
    print("\n✗ 분석할 데이터가 없습니다.")
    exit()

print(f"\n✓ 총 {len(combined_df)}개 데이터")
print(
    f"✓ 기간: {combined_df.index.min().strftime('%Y-%m-%d')} ~ {combined_df.index.max().strftime('%Y-%m-%d')}"
)

# ============================================================================
# 분석
# ============================================================================

print("\n" + "=" * 80)
print("데이터 분석")
print("=" * 80)

stats_summary = {}
growth_analysis = {}

print("\n[키워드별 통계]")
for keyword in combined_df.columns:
    data = combined_df[keyword].dropna()

    if len(data) > 0:
        avg = round(data.mean(), 2)
        max_val = round(data.max(), 2)
        min_val = round(data.min(), 2)
        std = round(data.std(), 2)

        growth = 0
        if len(data) > 1:
            mid = len(data) // 2
            first_half = data.iloc[:mid].mean()
            second_half = data.iloc[mid:].mean()
            if first_half > 0:
                growth = round(((second_half - first_half) / first_half) * 100, 2)

        stats_summary[keyword] = {
            "Average": avg,
            "Max": max_val,
            "Min": min_val,
            "Std": std,
            "Growth": growth,
        }
        growth_analysis[keyword] = growth

        print(f"\n• {keyword}")
        print(f"    Average: {avg}")
        print(f"    Max: {max_val}")
        print(f"    Growth: {growth:+.2f}%")

ranking = sorted(stats_summary.items(), key=lambda x: x[1]["Average"], reverse=True)
print("\n[Keyword Ranking]")
for idx, (keyword, stats) in enumerate(ranking, 1):
    print(f"  {idx}. {keyword}: {stats['Average']}")

# ============================================================================
# 시각화
# ============================================================================

print("\n" + "=" * 80)
print("시각화 생성")
print("=" * 80)

colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

# 1. 시계열
fig, ax = plt.subplots(figsize=(16, 8))

for idx, keyword in enumerate(combined_df.columns):
    ax.plot(
        combined_df.index,
        combined_df[keyword],
        marker="o",
        label=keyword,
        linewidth=2.5,
        markersize=5,
        alpha=0.8,
        color=colors[idx],
    )

ax.set_title(
    "US Diet Probiotics Keyword Trends (Google Trends)",
    fontsize=20,
    fontweight="bold",
    pad=20,
)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Search Interest (0-100)", fontsize=14)
ax.legend(loc="best", fontsize=12, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle="--")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

save_path = os.path.join(output_folder, "01_US_Trends_Timeseries.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✓ 01_US_Trends_Timeseries.png")
plt.close()

# 2. 성장률 (대안: 숫자를 막대 안쪽에 배치)
fig, ax = plt.subplots(figsize=(14, 9))

keywords_list = list(growth_analysis.keys())
values = list(growth_analysis.values())
colors_bar = ["#2ECC71" if v > 0 else "#E74C3C" for v in values]

bars = ax.barh(
    keywords_list, values, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=1.5
)

# 숫자를 막대 안쪽에 배치
for idx, (bar, value) in enumerate(zip(bars, values)):
    if abs(value) > 2:  # 막대가 충분히 길면 안쪽에
        ax.text(
            value * 0.5,
            idx,
            f"{value:+.1f}%",
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
    else:  # 막대가 짧으면 바깥쪽에
        if value > 0:
            ax.text(
                value + 0.5,
                idx,
                f"{value:+.1f}%",
                va="center",
                ha="left",
                fontsize=13,
                fontweight="bold",
            )
        else:
            ax.text(
                value - 0.5,
                idx,
                f"{value:+.1f}%",
                va="center",
                ha="right",
                fontsize=13,
                fontweight="bold",
            )

ax.set_xlabel("Growth Rate (%)", fontsize=14, fontweight="bold")
ax.set_title(
    "Keyword Growth Rate (First Half vs Second Half)",
    fontsize=18,
    fontweight="bold",
    pad=20,
)
ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
ax.grid(axis="x", alpha=0.3)

plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.08)

save_path = os.path.join(output_folder, "02_Growth_Rate_Analysis.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✓ 02_Growth_Rate_Analysis.png")
plt.close()


# 3. 평균 비교
fig, ax = plt.subplots(figsize=(14, 8))

keywords_avg = [stats["Average"] for stats in stats_summary.values()]
keywords_names = list(stats_summary.keys())

bars = ax.bar(
    range(len(keywords_names)),
    keywords_avg,
    color=colors[: len(keywords_names)],
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)

for idx, (bar, value) in enumerate(zip(bars, keywords_avg)):
    ax.text(
        idx,
        value + 1,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax.set_xticks(range(len(keywords_names)))
ax.set_xticklabels(keywords_names, rotation=45, ha="right", fontsize=11)
ax.set_ylabel("Average Search Interest", fontsize=14, fontweight="bold")
ax.set_title(
    "Average Search Interest by Keyword", fontsize=18, fontweight="bold", pad=20
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

save_path = os.path.join(output_folder, "03_Keyword_Comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✓ 03_Keyword_Comparison.png")
plt.close()

# 4. 박스플롯
fig, ax = plt.subplots(figsize=(14, 8))

data_for_box = [combined_df[col].dropna().values for col in combined_df.columns]
bp = ax.boxplot(
    data_for_box,
    labels=combined_df.columns,
    patch_artist=True,
    notch=True,
    showmeans=True,
)

for idx, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(colors[idx])
    patch.set_alpha(0.7)

ax.set_ylabel("Search Interest", fontsize=14, fontweight="bold")
ax.set_title(
    "Search Interest Distribution by Keyword", fontsize=18, fontweight="bold", pad=20
)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

save_path = os.path.join(output_folder, "04_Distribution_Boxplot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✓ 04_Distribution_Boxplot.png")
plt.close()

# 5. 인터랙티브
fig = go.Figure()

for idx, keyword in enumerate(combined_df.columns):
    fig.add_trace(
        go.Scatter(
            x=combined_df.index,
            y=combined_df[keyword],
            mode="lines+markers",
            name=keyword,
            line=dict(width=3, color=colors[idx]),
            marker=dict(size=7),
        )
    )

fig.update_layout(
    title={
        "text": "미국 다이어트 유산균 트렌드 분석 (인터랙티브)",
        "x": 0.5,
        "xanchor": "center",
        "font": {"size": 22},
    },
    xaxis_title="날짜",
    yaxis_title="검색 관심도 (0-100)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
)

save_path = os.path.join(output_folder, "05_Interactive_Chart.html")
fig.write_html(save_path)
print(f"✓ 05_Interactive_Chart.html")

# ============================================================================
# 결과 저장
# ============================================================================

print("\n" + "=" * 80)
print("결과 저장")
print("=" * 80)

# 1. 통합 데이터
save_path = os.path.join(output_folder, "Result01_Combined_Data.csv")
combined_df.to_csv(save_path, encoding="utf-8-sig")
print(f"✓ Result01_Combined_Data.csv")

# 2. 통계
save_path = os.path.join(output_folder, "Result02_Statistics.csv")
stats_df = pd.DataFrame(stats_summary).T
stats_df.to_csv(save_path, encoding="utf-8-sig")
print(f"✓ Result02_Statistics.csv")

# 3. 성장률
save_path = os.path.join(output_folder, "Result03_Growth_Analysis.csv")
growth_df = pd.DataFrame(
    {
        "Keyword": list(growth_analysis.keys()),
        "Growth_Rate(%)": list(growth_analysis.values()),
    }
).sort_values("Growth_Rate(%)", ascending=False)
growth_df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"✓ Result03_Growth_Analysis.csv")

# 4. 보고서
save_path = os.path.join(output_folder, "Result04_Final_Report.txt")
with open(save_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("미국 다이어트 유산균 트렌드 분석 보고서\n")
    f.write(f"분석 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}\n")
    f.write("=" * 80 + "\n\n")

    f.write("【1. 분석 개요】\n")
    f.write(f"• 분석 국가: 미국\n")
    f.write(f"• 분석 키워드: {len(available_files)}개\n")
    for keyword in available_files.keys():
        f.write(f"  - {keyword}\n")
    f.write(
        f"• 분석 기간: {combined_df.index.min().strftime('%Y-%m-%d')} ~ {combined_df.index.max().strftime('%Y-%m-%d')}\n"
    )
    f.write(f"• 데이터 포인트: {len(combined_df)}개\n\n")

    f.write("【2. 키워드 순위 (평균 검색 관심도)】\n")
    for idx, (keyword, stats) in enumerate(ranking, 1):
        f.write(f"{idx}위. {keyword}: {stats['Average']}\n")
    f.write("\n")

    f.write("【3. 성장률 분석 (전반기 대비 후반기)】\n")
    for keyword, rate in sorted(
        growth_analysis.items(), key=lambda x: x[1], reverse=True
    ):
        trend = "📈 증가" if rate > 0 else "📉 감소" if rate < 0 else "➡️ 유지"
        f.write(f"• {keyword}: {rate:+.2f}% {trend}\n")
    f.write("\n")

    f.write("【4. 키워드별 상세 통계】\n")
    for keyword, stats in stats_summary.items():
        f.write(f"\n{keyword}:\n")
        f.write(f"  - 평균: {stats['Average']}\n")
        f.write(f"  - 최대: {stats['Max']}\n")
        f.write(f"  - 최소: {stats['Min']}\n")
        f.write(f"  - 표준편차: {stats['Std']}\n")
        f.write(f"  - 성장률: {stats['Growth']:+.2f}%\n")
    f.write("\n")

    f.write("【5. 비엔날씬 신제품 개발 인사이트】\n\n")

    top_growth = max(growth_analysis.items(), key=lambda x: x[1])
    top_avg = max(stats_summary.items(), key=lambda x: x[1]["Average"])

    f.write(f"🔥 핵심 트렌드:\n")
    f.write(f"  • 가장 빠르게 성장: '{top_growth[0]}' ({top_growth[1]:+.1f}%)\n")
    f.write(f"  • 가장 높은 관심도: '{top_avg[0]}' (평균 {top_avg[1]['Average']})\n\n")


print(f"✓ Result04_Final_Report.txt")

# ============================================================================
# 완료
# ============================================================================

print("\n" + "=" * 80)
print("✅ 분석 완료!")
print("=" * 80)
print(f"\n결과 저장 위치: {output_folder}")
print("\n【생성된 파일】")
print("\n  [시각화]")
print("    • 01_US_Trends_Timeseries.png")
print("    • 02_Growth_Rate_Analysis.png")
print("    • 03_Keyword_Comparison.png")
print("    • 04_Distribution_Boxplot.png")
print("    • 05_Interactive_Chart.html")
print("\n  [분석 결과]")
print("    • Result01_Combined_Data.csv")
print("    • Result02_Statistics.csv")
print("    • Result03_Growth_Analysis.csv")
print("    • Result04_Final_Report.txt")
print("\n🎓 과제 작성에 활용하세요!")
print("=" * 80)
