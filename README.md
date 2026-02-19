# RomRadar

A data science tool that parses exported Telegram group chats to run **aspect-based sentiment analysis** on Android custom ROMs — helping you cut through the noise and figure out which ROM the community actually likes right now.

Built for the **Poco F5 / Redmi Note 12 Turbo** community chat, but works on any Telegram group export.

---

## What it does

- **Counts mentions** of every ROM in the chat — raw popularity signal
- **VADER sentiment analysis** per message — positive vs negative community reaction
- **Aspect-based breakdown** — separate scores for Battery, Performance, Stability, Camera, UI, Gaming, and Banking/DRM (keybox, Play Integrity, Widevine)
- **Temporal trend analysis** — weekly mention count + sentiment per ROM, so you can see if a ROM is getting better or worse _right now_
- **Recommendation / warning detection** — flags messages that explicitly recommend or warn against a ROM
- **Language detection** — reports what % of each ROM's mentions are in English, Hindi, or other languages (VADER is English-trained, so this flags potentially unreliable scores)
- **4 charts** — scatter plot, colour-coded bar chart, weekly trend lines, aspect heatmap
- **Self-contained HTML report** — `report.html` with all charts embedded, opens in any browser
- **CSV export** — `matched_messages.csv` with every matched message, date, sentiment score, and language code

---

## Covered ROMs

| ROM | Patterns matched |
|-----|-----------------|
| NeotericOS | neotericos, neoteric os, neoteric |
| crDroid | crdroid |
| InfinityX | infinityx, infinity x, project infinityx |
| EvolutionX | evolutionx, evolution x, evox |
| PixelOS | pixelos, pixel os, pos* |
| LineageOS | lineageos, lineage os, lineage, los* |
| ColorOS | coloros, color os, cos* |
| LunarisAOSP | lunarisaosp, lunaris aosp, lunaris |
| VoltageOS | voltageos, voltage os, voltage |
| AxionOS | axionos, axion os, axion |
| ZK UI | zkui, zk ui, zkos, zk os |
| HyperOS | hyperos, hyper os, hos* |
| MIUI | miui |
| OxygenOS | oxygenos, oos 14 |
| Neo | neo rom, neo* |
| ResuKI | resuki, resuki os |
| AlphaDroid | alphadroid |
| ParanoidOS | aospa, paranoid android |

\* Short slang patterns are context-filtered — only counted when the message also contains a ROM-related word (flash, rom, install, boot, keybox, etc.)

---

## Requirements

```
Python 3.9+
pandas
numpy
matplotlib
nltk
langdetect
```

Install everything at once:

```bash
pip install pandas numpy matplotlib nltk langdetect
```

---

## Setup

1. **Export your Telegram group chat**
   - Open Telegram Desktop → the group → ⋮ → Export Chat History
   - Format: **JSON**, uncheck all media
   - Save the file as `result.json` in the project folder

2. **Clone the repo**

   ```bash
   git clone https://github.com/rushdarshan/RomRadar.git
   cd RomRadar
   ```

3. **Run**

   ```bash
   python analyze.py
   ```

---

## Output

| File | Description |
|------|-------------|
| Terminal | Leaderboard, rec/warn counts, language % top-6 ROMs |
| `report.html` | Full report — open in browser |
| `matched_messages.csv` | Every matched message with ROM, date, score, language |
| 4 chart windows | Scatter, bar, temporal trends, aspect heatmap |

`result.json` and `matched_messages.csv` are in `.gitignore` — your chat data stays local.

---

## Adding ROMs

Edit the `rom_keywords` dict at the top of `analyze.py`:

```python
'MyROM': [r'\bmyrom\b', r'\bmy rom\b'],
```

For short/ambiguous abbreviations add a `_Slang` entry — it will be automatically context-filtered:

```python
'MyROM_Slang': [r'\bmr\b'],
```

---

## How to read the results

| Signal | What it means |
|--------|--------------|
| High mentions + high sentiment | Popular **and** well-liked — safest daily driver |
| Low mentions + high sentiment | Hidden gem — small but satisfied user base |
| High mentions + low sentiment | Widely discussed but controversial — lots of bug reports |
| Aspect heatmap green | Community positive about that specific aspect |
| Aspect heatmap red | Complaints about that aspect |
| Rec count high | Explicitly recommended by users |
| Warn count high | Users actively telling others to avoid it |
