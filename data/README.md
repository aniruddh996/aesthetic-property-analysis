# Data

## Files

| File | Description | Rows | Size |
|------|-------------|------|------|
| `all_images_15_properties.csv` | Claude API scores for all 15 Alexander properties | 800 | ~500 KB |
| `sam_features.csv` | SAM-derived structural features | 800 | ~200 KB |
| `votes.tsv` | Scenic-or-Not human crowd ratings | 212,000+ | ~15 MB |

## Sources

- **Images:** [Geograph UK](https://www.geograph.org.uk) — downloaded via API
- **Human ratings:** [Scenic-or-Not](https://scenicornot.datasciencelab.co.uk) — crowd-sourced scenic beauty votes (1–10 scale)

## Download

The original images (800 × ~1MB each) are not included in this repo due to size.
Run `notebooks/01_claude_api_scoring.ipynb` to download them via the Geograph API.
