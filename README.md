# epoch-bench

Epoch AI's **Chess Puzzles** Inspect task, reconstructed so it can be audited with
`inspect_audit`.

- `bench/task/chess_puzzles/__init__.py` and `puzzle_generator.py` are Epoch's published
  source: https://gist.github.com/greg-burnham/fc1c6f9ed32989073f58321b656b7b05 (fetched
  2026-09-07). One deviation: the CSV path resolves next to the module rather than
  against the working directory, so the task loads as an installed package.
- `bench/model.py` stands in for Epoch's private module of that name. The extractor
  model is bound through the `grader` model role (default `google/gemini-2.0-flash-001`,
  which 25 of the 38 public logs used; 12 used `openai/gpt-5-mini-2025-08-07`).
- `puzzles.csv` is rebuilt from the public logs (id, FEN, target). All 38 logs carry the
  identical 100 samples.

Logs: https://epoch-benchmarks-production-public.s3.us-east-2.amazonaws.com/inspect_ai_logs/<id>.eval
(manifest in `audit/logs.csv`).


## BFCL (`bench/bfcl`)

`bench/task/bfcl/__init__.py` builds the Berkeley Function-Calling Leaderboard dataset with
inspect_evals' own loaders, converters, solver and scorer (`inspect_evals/bfcl`, pinned in
`pyproject.toml` under the `bfcl` extra), minus two gates that only matter for running a
model: the memory categories' snapshot check and the web-search categories' SerpAPI key
check. The items of every scored category can therefore be loaded and audited; V1-V3
attempts generated with it are identical to inspect_evals'. Install with `pip install
"epoch-bench[bfcl] @ git+<this repo>@<sha>"`. Samples carry `metadata.stratum`, the
leaderboard's weighting stratum (non_live_ast, live_ast, irrelevance, multi_turn, memory,
web_search).

## bench/simpleqa_verified_anti_abstention

Epoch AI's SimpleQA Verified variant, reconstructed from Epoch's 64 published leaderboard logs (private source `epoch-research/benchmarks` @ 0bdf247, task version 0): the `codelion/SimpleQA-Verified` mirror at a pinned revision with sample id = row + 1, the question followed by the anti-abstention instruction, Epoch's verbatim grader template with a forced `submit` tool returning CORRECT, INCORRECT or NOT_ATTEMPTED, and accuracy as the metric. The grader is bound through the `grader` model role (Epoch used gpt-5.6-terra at medium reasoning). Install with the `simpleqa` extra.
