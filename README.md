# finbert-tone-runner

Runs **FinBERT-tone** (`yiyanghkust/finbert-tone`) financial-sentiment analysis via GitHub Actions — part of a one-public-repo-per-tool pattern: each tool gets its own repo, the commands to execute live in a repository variable, runs are triggered manually, and outputs are collected as artifacts.

## How to run an analysis

1. **Edit the commands**: repo **Settings → Secrets and variables → Actions → Variables** → set `COMMANDS` to the shell commands you want executed, e.g.
   ```
   mkdir -p results
   set -o pipefail; python scripts/tone_sentiment.py --input samples/poul_headlines.txt 2>&1 | tee results/run.log
   ```
2. **Trigger manually**: **Actions** tab → **Run analysis** → **Run workflow** (or dispatch it through the GitHub API).
3. **Collect results**: download the `results` artifact from the finished run. It contains `sentiment.json` (per-item labels + probabilities, label counts, mean probabilities), `report.md`, and the run log.

## Design notes

- **Model**: `yiyanghkust/finbert-tone` — a FinBERT variant fine-tuned for tone (financial sentiment / tone classification). It is not the same model as `ProsusAI/finbert` used in `finbert-runner`; scores from the two runners are not directly interchangeable.
- **Label mapping**: the script reads the model's own `id2label` from its config and maps its labels to positive / negative / neutral. The raw labels and raw scores are also preserved in `sentiment.json` (`raw_scores`, `predicted_raw_label`) in case the model's label names differ.
- **Generic and multi-market**: `scripts/tone_sentiment.py` scores arbitrary English financial texts — any stock, any market, any news source. Nothing in the workflow or script is tied to one exchange; the POUL headlines in `samples/` are only a pilot input, copied unchanged from `moos-labs/finbert-runner`.
- **Cheap repeat runs**: the pip cache plus an `actions/cache` entry for `~/.cache/huggingface` (key `finbert-tone-model-v1`) mean later runs skip both the package downloads and the model download.
- **CPU only**: `requirements.txt` pulls torch from the PyTorch CPU wheel index (via `--extra-index-url https://download.pytorch.org/whl/cpu`).
- **English only**: FinBERT-tone was fine-tuned on English financial text. Arabic headlines must be translated before scoring (the pilot headlines are English renderings of Arabic news items).
- **Old-config workaround**: `yiyanghkust/finbert-tone` (last modified 2022) ships a `config.json` with no `model_type` key, so modern transformers rejects it via `AutoConfig` / a plain pipeline. The script loads the raw config, injects `model_type: "bert"`, and builds `BertForSequenceClassification` / `BertTokenizer` explicitly.
