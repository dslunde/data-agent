# Data Agent Project
## Overview
Create a chat-based agent in Python that can interact with the linked dataset and answer user questions. No GUI or web API required—CLI is fine. Use all the coding tools, LLMs and resources you want, there are no restrictions on what you are allowed to use to complete this project provided you can bring those same tools with you to SynMax should you be hired. The agent should handle questions from simple retrieval to advanced analysis, including:

Pattern recognition (e.g., clustering, correlations, trends)

Anomaly detection (e.g., outliers, rule violations)

Proposing plausible causes behind observed relationships (with caveats and evidence)

Dataset: https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view?usp=sharing
 Due: Friday, August 29, 2025 at 5:00 PM CST

Do not include the dataset file in your repo.

## What to Build
### Capabilities
Your agent should:

Ingest the dataset (infers schema & types; handles missing values).

Understand natural-language questions.

Plan & execute analysis over the data.

Return a concise answer plus supporting evidence (methods used, selected columns, filters).

Handle both deterministic queries (e.g., “count of … in 2024”) and analytic tasks (patterns, anomalies, causal hypotheses with limitations).

### Engineering Requirements
Language: Python 3.10+

No front end or API required.

LLM options are OpenAI & Anthropic

Do not upload the dataset to GitHub. Your code should either:

- Prompt for a local path, or
- Download from the link at runtime and store under ./data/ (ensure it’s .gitignore’d).

## Setup & Keys
User will provide their oown keys via environment variables.

### Environment variables (use whichever are supported):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY

### Include README sections:
- Installation & quick start
- How to supply the dataset path or enable auto-download
- Example queries/outputs (anything your impressed with and want the user to see)
- Any assumptions or limitations

Dockerization is **not** required.

## Deliverables
Public GitHub repo link (code only, no dataset).
- README.md with clear installation & usage.
- requirements.txt


## Evaluation
The user will run a set of pre-written queries. Each query is scored on:

### Accuracy (70%)
- Correctness of numbers/tables/claims
- Sound methodology for patterns & anomalies
- Reasonable, evidence-backed causal hypotheses (not assertions)

### Speed (30%)
- Lower latency is better (measured per query)

### Bonus: extra credit for particularly insightful or actionable findings, e.g.,

- Surfacing non-obvious segments/clusters with business interpretation,
- Detecting data quality issues that change conclusions,
- Identifying potential confounders or validating with simple robustness checks.
