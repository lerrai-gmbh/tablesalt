# TableSalt — Robust Table Recognition from Degraded Scans

**Turn a photo or scan of a table into a structured table a program can query — even when the
image is skewed, shadowed, blurred or faxed.**

You get a data generator with exact ground truth, a scorer, two baselines and a working
submission. You hand in a Docker container; we run it offline on a hidden test set and rank all
teams with one deterministic command. No jury, no judge model.

Contents: [1 Problem](#1-the-problem) · [2 Task](#2-your-task) · [3 Getting started](#3-getting-started)
· [4 Data](#4-the-data) · [5 Output format](#5-output-format) · [6 Scoring](#6-scoring)
· [7 Submission](#7-submission) · [8 Machine and limits](#8-scoring-machine-and-limits)
· [9 Baselines](#9-baselines) · [10 Model leaderboard](#10-model-leaderboard-on-mlx-all-teams)
· [11 Rules](#11-rules) · [12 Ideas](#12-where-to-start) · [13 FAQ](#13-faq)
· [References](#references)

## 1. The problem

Table recognition on clean, digital-born documents works well: open models reach high scores on
standard benchmarks such as PubTabNet [1]. Real documents are rarely clean. They arrive as office
scans, phone photos and faxes, and table recognition degrades on exactly those inputs [3, 5].
Three things make it hard:

- **Degradation.** Skew, perspective, shadows, blur, noise, JPEG artifacts, fax binarization.
- **Structure.** Merged cells (row and column spans), two-row headers and borderless layouts where
  no line tells you where a column ends [2, 4, 5].
- **Content.** Numbers in mixed formats (`1,234.56`, `1.234,56`, `12 345`, `(1,234.56)`) where a
  single misread character changes the meaning.

A pipeline that gets the structure wrong silently attaches a number to the wrong row or column —
and whatever consumes the table downstream (a search index, a spreadsheet, an LLM) never notices.

## 2. Your task

Build a pipeline: **table image in → structured table out** (every cell with its text, row,
column and spans). Maximize accuracy, above all on the degraded images.

Anything that runs offline on the scoring machine (§8) is allowed: classic computer vision,
open-weight OCR models and vision-language models, fine-tuning, synthetic training data,
heuristics. Closed-model APIs are not — the official run has no network.

Runtime and image size are recorded and shown next to the ranking but are not part of the score.
Among similar scores, the leaner pipeline is the more useful one.

## 3. Getting started

**1. Create your team repository.** On this repository's GitHub page click **Use this template →
Create a new repository** and make it **private**. (Do not fork: a fork of a public repository is
public, so every other team could read your work.) Clone it.

**2. Install.** Python 3.10 or newer, plus Docker. For the Tesseract baseline also install
Tesseract and make sure `tesseract` is on your `PATH`:

| OS | Tesseract |
|---|---|
| Windows | installer from the UB Mannheim Tesseract builds, then add its folder to `PATH` |
| macOS | `brew install tesseract` |
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |

```
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt pytesseract
```

**3. Check that everything works** (about a minute):

```
python selftest_workflow.py --gt data/dev                                 # ends with ALL INVARIANTS HOLD
python baselines/baseline_tesseract.py --gt data/dev --out predictions/   # the floor baseline
python score_all.py --gt data/dev --pred predictions/                     # official score: FINAL ≈ 0.35
python evaluate.py  --gt data/dev --pred predictions/                     # per-table detail
```

**4. Look at the data.** Open a few `data/dev/table_*/image.*` next to their `gt/table.html`
(any browser renders it), especially the T2 and T3 ones.

**5. Build your pipeline**, write predictions for `data/dev`, score them with `score_all.py`.
Generate more data for training and validation whenever you need it (§4).

**6. Package it** in `submission_template/` and test the container exactly as we will run it (§7).

### What is in this repository

| Path | What it is |
|---|---|
| `data/dev/` | the dev set: 24 tables with ground truth |
| `generate_dataset.py` | the generator — the hidden test set comes from this exact code |
| `score_all.py` | the official score (final score, per tier, QA) |
| `evaluate.py` | TEDS, structure-only TEDS and cell F1, per table |
| `eval_qa.py` | the question-answering check |
| `selftest_workflow.py` | proves the scorer behaves (perfect = 1.0, damage scores lower, junk = 0) |
| `flatten_dataset.py` | copies a dataset's images into one folder, as your container receives them |
| `baselines/baseline_tesseract.py` | the floor: Tesseract + clustering words into rows and columns |
| `baselines/baseline_paddleocr.py` | a stronger reference: PaddleOCR's table pipeline, untuned |
| `submission_template/` | a working submission (Dockerfile + `predict.py`) running the Tesseract floor |

## 4. The data

All data is synthetic, so the ground truth is exact. `generate_dataset.py` draws a random table,
renders it as an image and damages it. Every table has a **style** and a **degradation tier**:

| Style | |
|---|---|
| `grid` | all cell borders |
| `hlines` | horizontal rules only (top, below the header, bottom) |
| `borderless` | no lines at all |

| Tier | Simulates | Damage |
|---|---|---|
| `T0` | clean document | none (PNG) |
| `T1` | office scanner | skew up to ±1.5°, light blur and noise, JPEG |
| `T2` | phone photo | skew up to ±4°, perspective, shadow across the page, blur, noise, JPEG |
| `T3` | fax / archive copy | skew, perspective, bleed-through from the back side, specks, black-and-white binarization, JPEG |

Every tier stays **legible**: a careful human can read every character, separator and sign. The
exact parameters are in `TIER_PARAMS` in `generate_dataset.py`.

Table content: 4–7 columns (the first holds row labels), 6–12 body rows, one or two header rows
(two-row headers group columns under a spanning label), occasional merged body cells
(`n/a`, `—`, `see note 3`), empty cells, units (`%`, `€`, `kg`, `ms`), English and German labels.
Each table uses **one** number format throughout:

| Format | Examples |
|---|---|
| English | `1,234.56` · `-1,234.56` or `(1,234.56)` |
| German | `1.234,56` · `-1.234,56` or `(1.234,56)` |
| Integer | `12,345` or `12 345` · `-12,345` or `(12,345)` |

**Dev set** (`data/dev/`, 24 tables, 6 per tier, 8 per style):

```
data/dev/manifest.json              one entry per table: id, image, style, tier, size
data/dev/table_000/image.png        the input (.png for T0, .jpg for T1-T3)
data/dev/table_000/gt/cells.json    ground truth in the submission format (§5)
data/dev/table_000/gt/table.html    the same as HTML with rowspan / colspan
data/dev/table_000/gt/table.md      markdown, only for tables without merged cells
data/dev/table_000/gt/qa.json       the questions used for the QA score (§6)
```

**More data, any amount** — the generator is yours to use for training and validation:

```
python generate_dataset.py --out data/train --n 5000 --seed 1
python generate_dataset.py --out data/val   --n 200  --seed 2
```

`--n` tables cycle through all tiers and styles. Git ignores everything under `data/` except
`data/dev/` (`.gitignore`), and the whole `data/` folder is left out of Docker builds
(`.dockerignore`), so generated data never bloats your repository or your image. Do not regenerate
`data/dev/` itself.

**Hidden test set:** about 96 tables from the same generator with a private seed, all tiers and
styles equally. Same distribution as the dev set — new tables.

**Fonts:** the dev and hidden sets are rendered with Arial. Windows and macOS have it; on Linux
the generator falls back to DejaVu Sans unless you point `TABLESALT_FONT` (and
`TABLESALT_FONT_BOLD`) at Arial `.ttf` files. Train on Arial if you can. Minimal Linux images and
containers have no fonts at all — install `fonts-dejavu-core` or set `TABLESALT_FONT`.

## 5. Output format

For every input image `<table_id>.png|.jpg`, write `<table_id>.json`: a list of cells.

```json
[{"text": "Position", "r": 0, "c": 0, "rs": 2, "cs": 1},
 {"text": "Delta",    "r": 0, "c": 1, "rs": 1, "cs": 3},
 {"text": "Jan",      "r": 1, "c": 1},
 {"text": "Feb",      "r": 1, "c": 2},
 {"text": "Mär",      "r": 1, "c": 3},
 {"text": "Cash flow","r": 2, "c": 0},
 {"text": "79.036,73 %", "r": 2, "c": 1}]
```

This is the top of `data/dev/table_000`: "Position" spans both header rows, "Delta" spans three
columns over "Jan", "Feb", "Mär".

- `r`, `c`: zero-based row and column of the cell's **top-left** grid position.
- `rs`, `cs`: row and column span, default 1.
- `text`: the cell text exactly as printed — keep the number format, signs, units. Whitespace is
  normalized when scoring; everything else counts.
- **Write every cell, including empty ones** (`"text": ""`). An empty cell is still a cell in the
  table structure; leaving it out costs TEDS.
- Header and body cells are not distinguished. Extra keys (like `header` in the ground truth) are
  ignored.

Alternatively write `<table_id>.html` with a single `<table>` using `rowspan` / `colspan`. If both
exist, `.json` wins. An unparseable file scores zero for that table.

## 6. Scoring

```
final = 0.85 × tier-weighted TEDS + 0.15 × QA accuracy
tier-weighted TEDS = 0.1·T0 + 0.2·T1 + 0.3·T2 + 0.4·T3      (mean TEDS of each tier)
tie-breaker: cell F1
```

`python score_all.py --gt <dataset> --pred <predictions>` computes exactly this.

**TEDS** (Tree-Edit-Distance-based Similarity [1]) compares your table with the true one as HTML
trees (`table → rows → cells`). It counts the cheapest edits that turn one into the other —
insert or delete a row or cell, change a span, fix a cell's text (in proportion to the characters
that differ) — and normalizes: 1 is a perfect table, 0 is nothing in common. A missing row or a
wrong span costs a lot; one wrong character in a long cell costs little.

**QA accuracy** checks that the table is usable, not just similar. Each table has about five
questions in `gt/qa.json`:

- *lookup*: `What is the value for "Steuern" in column "Q3"?`
- *aggregate*: `What is the sum of all numeric values in column "Q3" for the rows [...]?`

A script answers them **from your cells only**: it finds the row whose **first column** reads
exactly `Steuern` and the column whose header (anywhere in the **first three rows**) reads exactly
`Q3`, and takes the cell where they cross. Labels match ignoring case and repeated spaces; numbers
match by value, so `1,234.56`, `1.234,56` and `1234.56` are the same number. One wrong digit, a
column shifted by one, or a misread row label makes the answer wrong.

**Cell F1** (tie-breaker): share of cells with exactly the right text at exactly the right
position.

Degraded tiers weigh more on purpose: a pipeline that is perfect on clean images and fails on
faxes and photos reaches at most about 0.3.

## 7. Submission

You submit your **pipeline** — we build and run it. Your predictions files are not used.

**The contract**

- `submission_template/Dockerfile` in your repository, built with the **repository root** as
  context: `docker build -f submission_template/Dockerfile .`
- Its entrypoint takes two directories and nothing else:
  `<entrypoint> /data/images /data/predictions`
- `/data/images` holds one image per table, named `<table_id>.png` or `<table_id>.jpg`; write
  `/data/predictions/<table_id>.json` (§5) for each.

The template already does all of this with the Tesseract floor. Replace the internals of
`predict.py` (or the whole Dockerfile), keep the contract.

**Model weights.** The container runs **without network**, so the weights must be inside the
image. Download them during the build — the build has network — and do not commit them to git
(GitHub rejects files over 100 MB). For example:

```dockerfile
RUN pip install huggingface_hub && python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('<org>/<model>', revision='<commit hash>', local_dir='/app/weights')"
ENV HF_HUB_OFFLINE=1
```

Pin an exact revision: we build your image **after** the deadline, and a model that changed in
between would change your result. Do not keep weights under `data/` — it is excluded from the build.

**Test it exactly as we run it**

macOS / Linux:

```
python flatten_dataset.py data/dev flat
docker build -t my-tablesalt -f submission_template/Dockerfile .
docker run --rm --network none --gpus all -v "$PWD/flat:/data/images:ro" -v "$PWD/out:/data/predictions" my-tablesalt
python score_all.py --gt data/dev --pred out
```

Windows (PowerShell):

```
python flatten_dataset.py data/dev flat
docker build -t my-tablesalt -f submission_template/Dockerfile .
docker run --rm --network none --gpus all -v "${PWD}/flat:/data/images:ro" -v "${PWD}/out:/data/predictions" my-tablesalt
python score_all.py --gt data/dev --pred out
```

Drop `--gpus all` on a machine without an NVIDIA GPU. The template container scores a final 0.341
on the dev set this way. A container that does not build or run scores zero.

**Hand-in**

1. Invite `metinferhat@outlook.com` as a collaborator on your private repository.
2. Register your team name and repository URL with the organizers (details at kickoff).

## 8. Scoring machine and limits

Every submission runs on the same laptop, one at a time, offline:

| | |
|---|---|
| CPU | Intel Core Ultra 9 285H — the container gets 16 CPUs |
| RAM | 24 GB for the container |
| GPU | NVIDIA GeForce RTX 4070 Laptop, **8 GB VRAM**, driver 566.26 (CUDA up to 12.7) |
| Platform | Docker Desktop, `linux/amd64`; your image is built on this machine from your repository |
| Network | none at run time (`--network none`); available during the build |
| Time limit | 5 minutes startup + 15 seconds per table (about 29 minutes for 96 tables) |
| Build limit | 60 minutes |
| Image size | at most 15 GB |

The GPU is passed to every container; using it is optional. Building on Apple Silicon? Make sure
your Dockerfile also builds for `linux/amd64` (`docker build --platform linux/amd64 ...`).
Predictions written before a timeout still count — write each file as soon as it is done, not all
at the end.

## 9. Baselines

**Floor** — `baselines/baseline_tesseract.py`: Tesseract 5.5, words clustered into rows and
columns by position, no merged cells. Every team should beat it. Dev set:

| | T0 | T1 | T2 | T3 | weighted |
|---|---|---|---|---|---|
| TEDS | 0.54 | 0.46 | 0.26 | 0.35 | 0.362 |
| QA accuracy | 0.50 | 0.47 | 0.03 | 0.07 | 0.267 |
| **Final** | | | | | **0.348** |

Measured with Tesseract 5.5.0 installed on Windows; the template container (Tesseract 5.5.0 from
Debian) scores 0.341 — Tesseract builds differ slightly, so expect ±0.01 on your machine. It falls
apart on `grid` tables and on photos. With 6 tables per tier the dev numbers are noisy; generate a
larger validation set for decisions.

**Reference** — `baselines/baseline_paddleocr.py`: PaddleOCR's stock table-recognition pipeline,
no tuning. A much stronger starting point; see the script's header for installation (about
1.3 GB of weights). Run it on the dev set to see where it breaks.

## 10. Model leaderboard on MLX (all teams)

Before the challenge we ran every open-weight OCR and document model in our registry — 26 models up
to 15 B — through the official scorer on the dev set, **locally on Apple Silicon with MLX** instead
of a GPU. The board is published for all teams: it shows what this data allows on a laptop, which
tier breaks which model, and that 1 B models are competitive with 9 B ones.

| | |
|---|---|
| Models | 22 scored, open weights up to 15 B, 4–8 bit MLX builds (`mlx-vlm`) |
| Data | `data/dev` — 24 tables, T0–T3 |
| Score | the official objective, unchanged: S = 0.85 · TEDS_w + 0.15 · QA, TEDS_w = 0.10·T0 + 0.20·T1 + 0.30·T2 + 0.40·T3, tie-break CellF1 |
| Speed | seconds per table, one stream, Metal on the host — not the scoring machine |

| # | Model | Params | Quant | T0 | T1 | T2 | T3 | QA | CellF1 | FINAL | s/table |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MinerU2.5 1.2B | 1.20B | bf16 | 0.959 | 0.992 | 0.956 | 0.966 | 0.958 | 0.937 | **0.9659** | 2.1 |
| 2 | Chandra OCR 2 | 9.00B | 8bit | 0.932 | 0.990 | 0.961 | 0.960 | 0.967 | 0.975 | **0.9640** | 14.9 |
| 3 | PaddleOCR-VL 0.9B | 0.90B | 8bit | 0.941 | 0.984 | 0.961 | 0.955 | 0.933 | 0.949 | **0.9570** | 1.5 |
| 4 | dots.ocr 3B | 3.00B | 8bit | 0.910 | 0.937 | 0.927 | 0.942 | 1.000 | 0.958 | **0.9432** | 6.1 |
| 5 | Qwen3.5 9B | 9.00B | 4bit | 0.941 | 0.941 | 0.943 | 0.941 | 0.875 | 0.883 | **0.9315** | 18.5 |
| 6 | dots.mocr 3B | 3.00B | 8bit | 0.927 | 0.965 | 0.908 | 0.945 | 0.883 | 0.950 | **0.9280** | 7.7 |
| 7 | olmOCR-2 7B | 7.00B | 8bit | 0.834 | 0.923 | 0.928 | 0.950 | 0.933 | 0.927 | **0.9274** | 23.2 |
| 8 | GLM-4.6V-Flash 9B | 9.00B | 4bit | 0.909 | 0.933 | 0.913 | 0.959 | 0.883 | 0.882 | **0.9274** | 10.6 |
| 9 | LightOnOCR-3 0.8B | 0.80B | 8bit | 0.887 | 0.947 | 0.914 | 0.939 | 0.858 | 0.814 | **0.9175** | 3.4 |
| 10 | Qwen3-VL 4B Instruct | 4.00B | 4bit | 0.922 | 0.950 | 0.895 | 0.936 | 0.825 | 0.895 | **0.9100** | 8.4 |
| 11 | Nanonets-OCR2 3B | 3.00B | 8bit | 0.897 | 0.923 | 0.906 | 0.933 | 0.850 | 0.881 | **0.9087** | 9.8 |
| 12 | Qwen3-VL 8B Instruct | 8.00B | 4bit | 0.915 | 0.948 | 0.945 | 0.935 | 0.742 | 0.918 | **0.9087** | 13.5 |
| 13 | Qwen2.5-VL 7B | 8.00B | 4bit | 0.882 | 0.906 | 0.913 | 0.926 | 0.475 | 0.753 | **0.8479** | 11.3 |
| 14 | InternVL3 8B | 8.00B | 6bit | 0.882 | 0.878 | 0.855 | 0.860 | 0.708 | 0.709 | **0.8409** | 17.4 |
| 15 | MiniCPM-V 4.6 | 1.00B | 4bit | 0.872 | 0.871 | 0.689 | 0.943 | 0.700 | 0.771 | **0.8235** | 3.1 |
| 16 | LFM2.5-VL 3B | 3.00B | 4bit | 0.829 | 0.864 | 0.805 | 0.866 | 0.483 | 0.540 | **0.7896** | 3.9 |
| 17 | Ministral 3 14B (8B build) | 14.00B | 4bit | 0.881 | 0.828 | 0.863 | 0.734 | 0.583 | 0.614 | **0.7728** | 23.5 |
| 18 | Gemma 4 12B | 12.00B | 4bit | 0.609 | 0.765 | 0.770 | 0.760 | 0.642 | 0.651 | **0.7330** | 30.7 |
| 19 | Pixtral 12B | 12.00B | 4bit | 0.780 | 0.743 | 0.865 | 0.782 | 0.242 | 0.550 | **0.7154** | 18.9 |
| 20 | Molmo2 8B | 8.00B | 4bit | 0.689 | 0.838 | 0.637 | 0.537 | 0.233 | 0.277 | **0.5810** | 26.4 |
| 21 | SmolDocling 256M | 0.26B | bf16 | 0.512 | 0.559 | 0.657 | 0.431 | 0.283 | 0.362 | **0.4953** | 11.5 |
| 22 | Granite-Docling 258M | 0.26B | bf16 | 0.801 | 0.892 | 0.443 | 0.309 | 0.258 | 0.386 | **0.4766** | 3.7 |

**Reading it.** PaddleOCR-VL 0.9B is third overall at 1.5 s per table — the fastest of the top
three, and a specialist pipeline rather than a general chat model. MinerU2.5 1.2B beats every 9B
model here. CellF1 and QA separate models with similar TEDS: Qwen3-VL 8B keeps the structure
(CellF1 0.918) but answers only 0.742 of the questions, Pixtral 12B reads cells at 0.550 with 0.242
QA. The top three hold ≥ 0.93 on every tier, while below them the damage concentrates in single
tiers: MiniCPM-V 4.6 drops to 0.689 in T2 (phone photos), Granite-Docling to 0.309 and SmolDocling
to 0.431 in T3 (fax), Pixtral 12B to 0.743 in T1.

**MLX is not the scoring machine.** The official run is CUDA on an 8 GB RTX 4070 at 15 s per table,
with other quantisations and kernels, so the third decimal and the seconds per table will not carry
over. The ranking, and the shape of the failures, do.

Not scored: DeepSeek-OCR 3B and DeepSeek-OCR-2 3B (the MLX builds carry no `processor_class`),
GOT-OCR2.0 0.58B (needs torch and verovio in the server), Phi-4-multimodal 14B (weights do not match
the mlx-vlm definition). They are not zeros — they simply did not run.

## 11. Rules

1. Open-weight models only at inference. Training and data synthesis beforehand are free —
   generated data and public table datasets such as PubTabNet [1], PubTables-1M [2] or
   FinTabNet [4] included.
2. Only what your container produces in the official run counts.
3. Do not try to detect or special-case the evaluation environment, and do not read anything but
   `/data/images`.
4. Any hardware during development; only the official run is scored.

## 12. Where to start

- **Look at your errors first.** `evaluate.py` prints TEDS per table; compare the worst ones with
  their images.
- **Pre-processing:** deskew, perspective correction, illumination correction, denoising. The
  degradations are known — `TIER_PARAMS` tells you what you are up against.
- **Structure, then text:** detect rows, columns and merged cells (e.g. Table Transformer [2]),
  then read each cell crop with an OCR model.
- **Strong open models:** PaddleOCR's table pipeline, OCR and document vision-language models.
  Mind the 8 GB of VRAM and the time limit.
- **Post-processing:** a table uses one number format — use the cells you read well to fix
  separators and signs in the ones you did not.
- **Fine-tune on generated data:** unlimited, exactly labeled, in the test distribution.

## 13. FAQ

**Can we train on the dev set?** Yes, but you lose your only fixed yardstick. Generate training
data instead and keep `data/dev` for comparison.

**Can we change the generator?** Locally, as much as you like — more variety can help. The hidden
set is generated with the unchanged code in this repository.

**JSON or HTML output?** Same score. JSON is simpler to get right.

**Does the GPU help?** Only if your pipeline uses it. CPU-only pipelines are welcome; runtime is
reported next to the ranking.

**Can we submit several times?** Push as often as you like; only the last commit before the
deadline counts.

**Something is broken or unclear?** Ask the organizers. Fixes to the kit will be announced.

## References

[1] X. Zhong, E. ShafieiBavani, A. Jimeno Yepes. *Image-based table recognition: data, model, and
    evaluation.* ECCV 2020. arXiv:1911.10683 — PubTabNet and TEDS.
[2] B. Smock, R. Pesala, R. Abraham. *PubTables-1M: Towards comprehensive table extraction from
    unstructured documents.* CVPR 2022 — Table Transformer (TATR).
[3] L. Fu et al. *OCRBench v2: An improved benchmark for evaluating large multimodal models on
    visual text localization and reasoning.* 2025. arXiv:2501.00321
[4] X. Zheng et al. *Global Table Extractor (GTE): A framework for joint table identification and
    cell structure recognition using visual context.* WACV 2021 — FinTabNet.
[5] R. Long et al. *Parsing table structures in the wild.* ICCV 2021 — WTW, photographed tables.

## License

MIT — see [LICENSE](LICENSE).
