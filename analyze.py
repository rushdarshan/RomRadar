import json
import re
import io
import base64
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
from langdetect.detector_factory import DetectorFactory

nltk.download('vader_lexicon', quiet=True)
DetectorFactory.seed = 0  # reproducible language detection

# ─────────────────────────── CONSTANTS ────────────────────────────────────────

rom_keywords = {
    # ── existing ROMs (updated patterns) ──────────────────────────────────────
    'NeotericOS':       [r'\bneotericos\b', r'\bneoteric os\b', r'\bneoteric\b'],
    'crDroid':          [r'\bcrdroid\b'],
    'InfinityX':        [r'\binfinityx\b', r'\binfinity x\b', r'\bproject infinityx\b'],
    'EvolutionX':       [r'\bevolutionx\b', r'\bevolution x\b', r'\bevox\b'],
    'PixelOS':          [r'\bpixelos\b', r'\bpixel os\b'],
    'PixelOS_Slang':    [r'\bpos\b'],          # filtered by context_words
    'LineageOS':        [r'\blineageos\b', r'\blineage os\b', r'\blineage\b'],
    'LineageOS_Slang':  [r'\blos\b'],          # filtered by context_words
    'ColorOS':          [r'\bcoloros\b', r'\bcolor os\b'],
    'ColorOS_Slang':    [r'\bcos\b'],          # filtered by context_words
    'LunarisAOSP':      [r'\blunarisaosp\b', r'\blunaris aosp\b', r'\blunaris\b'],
    'VoltageOS':        [r'\bvoltageos\b', r'\bvoltage os\b', r'\bvoltage\b'],
    'AxionOS':          [r'\baxionos\b', r'\baxion os\b', r'\baxion\b'],
    'ZK UI':            [r'\bzkui\b', r'\bzk ui\b', r'\bzkos\b', r'\bzk os\b'],
    # ── new ROMs identified in the Poco F5 chat ───────────────────────────────
    'HyperOS':          [r'\bhyperos\b', r'\bhyper os\b'],
    'HyperOS_Slang':    [r'\bhos\b'],          # filtered by context_words
    'MIUI':             [r'\bmiui\b'],
    'OxygenOS':         [r'\boxygenos\b', r'\boos 14\b'],
    'Neo':              [r'\bneo rom\b'],       # exact phrase; bare \bneo\b via Neo_Slang
    'Neo_Slang':        [r'\bneo\b'],           # filtered by context_words
    'ResuKI':           [r'\bresuki\b', r'\bresuki os\b'],
    'AlphaDroid':       [r'\balphadroid\b'],
    'ParanoidOS':       [r'\baospa\b', r'\bparanoid android\b'],
}

context_words = [
    'flash', 'rom', 'install', 'update', 'bug', 'battery',
    'sot', 'smooth', 'port', 'android', 'boot', 'recovery',
    'kernel', 'keybox', 'integrity',
]

ASPECT_KEYWORDS = {
    'battery':      [r'\bbattery\b', r'\bsot\b', r'\bdrain\b', r'\bcharging\b',
                     r'\bbatterylife\b', r'\bidle drain\b', r'\bstandby\b'],
    'performance':  [r'\bsmooth\b', r'\blag\b', r'\bfps\b', r'\bperformance\b',
                     r'\bfast\b', r'\bslow\b', r'\bthermal\b'],
    'stability':    [r'\bstable\b', r'\bstability\b', r'\bcrash\b', r'\bbug\b',
                     r'\bfreeze\b', r'\breboot\b', r'\bbootloop\b'],
    'camera':       [r'\bcamera\b', r'\bgcam\b', r'\bphoto\b', r'\bpicture\b'],
    'ui':           [r'\bui\b', r'\btheme\b', r'\bcustomization\b', r'\blook\b'],
    'gaming':       [r'\bgaming\b', r'\bgame\b', r'\bgames\b', r'\bpubg\b',
                     r'\bbgmi\b', r'\bgenshin\b', r'\bwuthering\b', r'\bthermal\b'],
    'banking_drm':  [r'\bkeybox\b', r'\bplay integrity\b', r'\bstrong integrity\b',
                     r'\bdevice integrity\b', r'\bnetflix\b', r'\bwidevine\b',
                     r'\bbanking\b', r'\bupi\b', r'\bpayments?\b'],
}

REC_KEYWORDS  = [
    r'\brecommend\b', r'\bbest rom\b', r'\bgo for\b',
    r'\bswitch to\b', r'\bworth\b', r'\bmust use\b', r'\btry.*rom\b',
    r'\bgood for daily\b', r'\bbest for gaming\b', r'\bbest gaming rom\b',
    r'\bdaily driver\b',
]
WARN_KEYWORDS = [
    r'\bavoid\b', r"\bdon'?t use\b", r'\bstay away\b',
    r'\bworst\b', r'\bnot good\b', r'\bnot worth\b',
    r'\bnot recommended\b', r'\bdo not flash\b',
]

# ─────────────────────────── HELPERS ──────────────────────────────────────────

def extract_text(message_text):
    if isinstance(message_text, str):
        return message_text
    elif isinstance(message_text, list):
        parts = []
        for p in message_text:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and 'text' in p:
                parts.append(p['text'])
        return ''.join(parts)
    return ''


def detect_language(text):
    """Return ISO 639-1 language code or 'unknown' for short/ambiguous text."""
    try:
        if len(text.strip()) < 20:
            return 'unknown'
        return detect(text)
    except LangDetectException:
        return 'unknown'


def analyze_aspects(text, sia):
    """Return {aspect: compound_score} for every aspect keyword found in text."""
    found = {}
    for aspect, patterns in ASPECT_KEYWORDS.items():
        if re.search('|'.join(patterns), text, re.IGNORECASE):
            found[aspect] = sia.polarity_scores(text)['compound']
    return found


def classify_intent(text):
    """Return (is_recommendation, is_warning) booleans."""
    is_rec  = bool(re.search('|'.join(REC_KEYWORDS),  text, re.IGNORECASE))
    is_warn = bool(re.search('|'.join(WARN_KEYWORDS), text, re.IGNORECASE))
    return is_rec, is_warn


def fig_to_base64(fig):
    """Render figure to base64 PNG string (does NOT close the figure)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ─────────────────────────── PLOT FUNCTIONS ───────────────────────────────────

def plot_scatter(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results_df['Mentions'], results_df['Average_Sentiment'],
               color='teal', s=100, alpha=0.7)
    for _, row in results_df.iterrows():
        ax.annotate(row['ROM'], (row['Mentions'], row['Average_Sentiment']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('ROM Analysis: Mentions vs. Overall Sentiment')
    ax.set_xlabel('Number of Mentions (Popularity)')
    ax.set_ylabel('Average Sentiment Score (-1 to 1)')
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    return fig


def plot_mentions_bar(results_df):
    """Horizontal bar chart — bars coloured by sentiment (green=positive, red=negative)."""
    df = results_df.sort_values('Mentions')
    norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    colors = plt.cm.RdYlGn(norm(df['Average_Sentiment'].clip(-0.3, 0.3)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['ROM'], df['Mentions'], color=colors)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel('Number of Mentions')
    ax.set_title('ROM Mentions  (bar colour = sentiment: green positive, red negative)')
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    fig.tight_layout()
    return fig


def plot_temporal_trends(temporal_data):
    """Weekly mention count (top) and sentiment (bottom) for ROMs with ≥ 2 weeks of data."""
    eligible = {rom: wdf for rom, wdf in temporal_data.items() if len(wdf) >= 2}
    if not eligible:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'Not enough weekly data', ha='center', va='center')
        ax.set_title('Weekly ROM Trends')
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    colors = plt.cm.tab10.colors

    for idx, (rom, wdf) in enumerate(eligible.items()):
        color = colors[idx % len(colors)]
        x = wdf['week_label']
        axes[0].plot(x, wdf['mentions_count'], marker='o', label=rom,
                     color=color, linewidth=1.5, markersize=4)
        axes[1].plot(x, wdf['avg_sentiment'], marker='o', label=rom,
                     color=color, linewidth=1.5, markersize=4)

    for ax, title in zip(axes, ['Weekly Mentions per ROM', 'Weekly Average Sentiment per ROM']):
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=3, loc='upper left')
        ax.tick_params(axis='x', rotation=40)
        ax.grid(True, linestyle=':', alpha=0.5)

    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('Avg Sentiment (-1 to 1)')
    axes[0].set_ylabel('Message Count')
    fig.tight_layout()
    return fig


def plot_aspect_heatmap(aspect_data_avg, results_df):
    """Heatmap: ROMs (rows) × aspects (columns). Grey = no data."""
    aspects = list(ASPECT_KEYWORDS.keys())
    rom_order = results_df['ROM'].tolist()

    matrix = np.full((len(rom_order), len(aspects)), np.nan)
    for i, rom in enumerate(rom_order):
        for j, asp in enumerate(aspects):
            val = aspect_data_avg.get(rom, {}).get(asp)
            if val is not None:
                matrix[i, j] = val

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad('lightgray')
    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(9, max(5, len(rom_order) * 0.6)))
    im = ax.imshow(masked, cmap=cmap, vmin=-0.5, vmax=0.5, aspect='auto')

    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels([a.capitalize() for a in aspects])
    ax.set_yticks(range(len(rom_order)))
    ax.set_yticklabels(rom_order)

    for i in range(len(rom_order)):
        for j in range(len(aspects)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7)

    plt.colorbar(im, ax=ax, label='Avg Sentiment')
    ax.set_title('Aspect Sentiment by ROM  (grey = no data for that aspect)')
    fig.tight_layout()
    return fig

# ─────────────────────────── HTML REPORT ──────────────────────────────────────

def generate_html_report(results_df, aspect_df, rec_df, lang_table,
                         chart_b64s, output_path='report.html'):
    css = """
    body { font-family: Arial, sans-serif; margin: 30px; background: #f7f7f7; color: #222; }
    h1   { color: #1a1a2e; }
    h2   { color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 4px; }
    .card { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 28px;
            box-shadow: 0 2px 8px rgba(0,0,0,.08); }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th    { background: #16213e; color: #fff; padding: 8px 12px; text-align: left; }
    td    { padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }
    tr:nth-child(even) td { background: #f5f5f5; }
    .charts { display: flex; flex-wrap: wrap; gap: 20px; }
    .charts img { max-width: 100%; border-radius: 6px; border: 1px solid #ddd; }
    .note { font-size: 12px; color: #888; margin-top: 6px; }
    """

    def df_to_html(df):
        return df.to_html(index=False, border=0, classes='')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    rec_note = ('Counts messages that contain recommendation keywords '
                '(recommend, best, switch to…) or warning keywords '
                '(avoid, worst, don\'t use…) alongside the ROM name.')
    lang_note = ('VADER sentiment is English-trained. Non-English mentions may '
                 'have unreliable scores. "unknown" = message too short to classify.')

    charts_html = ''.join(
        f'<img src="data:image/png;base64,{b64}" alt="{name}">'
        for name, b64 in chart_b64s.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Custom ROM Analysis Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Custom ROM Analysis Report</h1>
  <p style="color:#888">Generated: {timestamp} &nbsp;|&nbsp; Poco F5 / Redmi Note 12 Turbo group chat</p>

  <div class="card">
    <h2>Overall Leaderboard</h2>
    {df_to_html(results_df)}
  </div>

  <div class="card">
    <h2>Charts</h2>
    <div class="charts">{charts_html}</div>
  </div>

  <div class="card">
    <h2>Recommendation & Warning Counts</h2>
    {df_to_html(rec_df.sort_values('Recommendations', ascending=False))}
    <p class="note">{rec_note}</p>
  </div>

  <div class="card">
    <h2>Aspect Sentiment (per ROM)</h2>
    {df_to_html(aspect_df)}
    <p class="note">Scores are average VADER compound scores (-1 to +1) for messages
    mentioning each aspect alongside the ROM. Blank = no messages mentioned that aspect.</p>
  </div>

  <div class="card">
    <h2>Language Breakdown (% of matched messages)</h2>
    {df_to_html(lang_table)}
    <p class="note">{lang_note}</p>
  </div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report written  ->  {output_path}")

# ─────────────────────────── LOAD DATA ────────────────────────────────────────

print("Loading result.json …")
with open('result.json', 'r', encoding='utf-8') as file:
    chat_data = json.load(file)

messages = [
    {'date': msg.get('date'), 'text': extract_text(msg.get('text', '')).strip()}
    for msg in chat_data.get('messages', [])
    if msg.get('type') == 'message' and extract_text(msg.get('text', '')).strip()
]
df = pd.DataFrame(messages)
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df):,} messages  ({df['date'].min().date()} → {df['date'].max().date()})")

# ─────────────────────────── ROM MATCHING LOOP ────────────────────────────────

sia = SentimentIntensityAnalyzer()
results       = []
all_matched_rows = []
temporal_raw  = defaultdict(list)          # rom -> [DataFrames with date + sentiment]
aspect_raw    = defaultdict(lambda: defaultdict(list))  # rom -> aspect -> [scores]
rec_data      = defaultdict(lambda: {'rec': 0, 'warn': 0})
lang_raw      = defaultdict(list)          # rom -> [language codes]

print("Analysing ROM mentions …")
for rom_name, patterns in rom_keywords.items():
    combined_pattern = '|'.join(patterns)
    mentions = df[df['text'].str.contains(combined_pattern, case=False, na=False)].copy()

    # Generic slang context filter: any _Slang entry requires ROM context words
    if rom_name.endswith('_Slang'):
        ctx = '|'.join([rf'\b{w}\b' for w in context_words])
        mentions = mentions[mentions['text'].str.contains(ctx, case=False, na=False)]
        rom_name = rom_name.replace('_Slang', '')

    if mentions.empty:
        continue

    # Sentiment
    mentions['sentiment'] = mentions['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Language detection
    mentions['language'] = mentions['text'].apply(detect_language)
    lang_raw[rom_name].extend(mentions['language'].tolist())

    # Aspect analysis
    for text in mentions['text']:
        for asp, score in analyze_aspects(text, sia).items():
            aspect_raw[rom_name][asp].append(score)

    # Rec / warn
    for text in mentions['text']:
        is_rec, is_warn = classify_intent(text)
        if is_rec:  rec_data[rom_name]['rec']  += 1
        if is_warn: rec_data[rom_name]['warn'] += 1

    # Temporal
    temporal_raw[rom_name].append(mentions[['date', 'sentiment']].copy())

    # CSV rows
    mentions_csv = mentions[['date', 'text', 'sentiment', 'language']].copy()
    mentions_csv.insert(0, 'ROM', rom_name)
    all_matched_rows.append(mentions_csv)

    results.append({
        'ROM': rom_name,
        'Mentions': len(mentions),
        'Average_Sentiment': mentions['sentiment'].mean(),
    })

# ─────────────────────────── AGGREGATE ────────────────────────────────────────

results_df = pd.DataFrame(results)
if results_df.empty:
    print("No mentions found for the specified ROMs.")
    raise SystemExit

# Merge duplicate ColorOS rows (from main pattern + slang)
results_df['_weighted'] = results_df['Average_Sentiment'] * results_df['Mentions']
results_df = results_df.groupby('ROM', as_index=False).agg(
    Mentions=('Mentions', 'sum'),
    _weighted_sum=('_weighted', 'sum'),
)
results_df['Average_Sentiment'] = results_df['_weighted_sum'] / results_df['Mentions']
results_df = results_df.drop(columns='_weighted_sum')

results_df = results_df.sort_values(
    by=['Mentions', 'Average_Sentiment'], ascending=[False, False]
).reset_index(drop=True)

# Aspect averages
aspect_data_avg = {
    rom: {asp: round(sum(v) / len(v), 4) for asp, v in aspects.items() if v}
    for rom, aspects in aspect_raw.items()
}

# Language value_counts per ROM
lang_data = {
    rom: pd.Series(langs).value_counts(normalize=True).mul(100).round(1)
    for rom, langs in lang_raw.items()
}

# Temporal weekly dfs
temporal_data = {}
for rom, df_list in temporal_raw.items():
    combined = pd.concat(df_list, ignore_index=True)
    weekly = (
        combined.assign(week=combined['date'].dt.to_period('W'))
        .groupby('week')
        .agg(mentions_count=('sentiment', 'count'), avg_sentiment=('sentiment', 'mean'))
        .reset_index()
    )
    weekly['week_label'] = weekly['week'].apply(lambda p: p.start_time.strftime('%b %d'))
    temporal_data[rom] = weekly

# Rec / warn DataFrame
rec_df = pd.DataFrame([
    {'ROM': rom, 'Recommendations': d['rec'], 'Warnings': d['warn']}
    for rom, d in rec_data.items()
]).sort_values('Recommendations', ascending=False).reset_index(drop=True)

# Language table (top 5 languages per ROM)
lang_rows = []
for rom in results_df['ROM']:
    if rom in lang_data:
        row = {'ROM': rom}
        row.update(lang_data[rom].head(5).to_dict())
        lang_rows.append(row)
lang_table = pd.DataFrame(lang_rows).fillna(0.0)

# Aspect table
aspects_list = list(ASPECT_KEYWORDS.keys())
aspect_rows = []
for rom in results_df['ROM']:
    row = {'ROM': rom}
    row.update({
        asp.capitalize(): aspect_data_avg.get(rom, {}).get(asp)
        for asp in aspects_list
    })
    aspect_rows.append(row)
aspect_df = pd.DataFrame(aspect_rows)

# ─────────────────────────── TERMINAL OUTPUT ──────────────────────────────────

print("\n─── CUSTOM ROM LEADERBOARD ──────────────────────────────────")
print(results_df.to_string(index=False))

print("\n─── RECOMMENDATION / WARNING COUNTS ────────────────────────")
print(rec_df.to_string(index=False))

print("\n─── LANGUAGE BREAKDOWN (top 3 langs, % of matched messages) ─")
for rom in results_df['ROM'].head(6):
    if rom in lang_data:
        top3 = lang_data[rom].head(3).to_dict()
        print(f"  {rom:<14}: {top3}")

# ─────────────────────────── CSV EXPORT ───────────────────────────────────────

if all_matched_rows:
    export_df = pd.concat(all_matched_rows, ignore_index=True)
    export_df = export_df.sort_values(['ROM', 'date']).reset_index(drop=True)
    csv_path = 'matched_messages.csv'
    export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nExported {len(export_df):,} matched messages  ->  {csv_path}")

# ─────────────────────────── CHARTS + HTML ────────────────────────────────────

print("\nBuilding charts …")
figs = {}
figs['scatter']  = plot_scatter(results_df)
figs['mentions'] = plot_mentions_bar(results_df)
figs['temporal'] = plot_temporal_trends(temporal_data)
figs['heatmap']  = plot_aspect_heatmap(aspect_data_avg, results_df)

# Encode to base64 first so HTML is written before plt.show() blocks
chart_b64s = {name: fig_to_base64(fig) for name, fig in figs.items()}

generate_html_report(
    results_df=results_df,
    aspect_df=aspect_df,
    rec_df=rec_df,
    lang_table=lang_table,
    chart_b64s=chart_b64s,
)

print("\nDone. Displaying charts (close windows to exit) …")
plt.show()
