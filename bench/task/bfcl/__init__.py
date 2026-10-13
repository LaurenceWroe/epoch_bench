"""BFCL as inspect_evals runs it, loadable for auditing without the agentic gates.

`inspect_evals/bfcl` refuses to load the memory categories unless a `bfcl_prereqs`
snapshot for the same model exists on this machine, and silently drops the web-search
categories when `SERPAPI_API_KEY` is unset. Both gates concern *running* a model: the
items themselves (questions, tool docs, reference answers) load without either. An
audit reads items, so this task builds the same dataset with the same converters,
solver and scorer as inspect_evals, minus the two gates. Attempts generated with it on
the V1-V3 categories are identical to inspect_evals' own; the agentic categories can be
loaded and audited here, and can be run only when the upstream prerequisites are met.

Everything the task does is delegated to inspect_evals' own functions at the pinned
revision in pyproject.toml, so the items, tools, scoring and version are theirs.
"""

from __future__ import annotations

import logging
import shutil

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

logger = logging.getLogger(__name__)

# the categories the official leaderboard scores, grouped by its weighting strata
STRATA: dict[str, list[str]] = {
    "non_live_ast": [
        "simple_python",
        "simple_java",
        "simple_javascript",
        "multiple",
        "parallel",
        "parallel_multiple",
    ],
    "live_ast": ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"],
    "irrelevance": ["irrelevance", "live_irrelevance", "live_relevance"],
    "multi_turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
    "memory": ["memory_kv", "memory_vector", "memory_rec_sum"],
    "web_search": ["web_search_base", "web_search_no_snippet"],
}
SCORED_CATEGORIES: list[str] = [c for names in STRATA.values() for c in names]


@task
def bfcl(
    categories: str | list[str] | None = None,
    snapshot_id: str | None = None,
) -> Task:
    """BFCL items for every scored category, built with inspect_evals' own loaders.

    Args:
        categories: Category names or the aliases inspect_evals accepts, plus the
            stratum names in STRATA. Defaults to every category the leaderboard scores.
        snapshot_id: Memory snapshot to read when memory categories are *run*; not
            required to load them.
    """
    from inspect_evals.bfcl.backends import download_func_source, setup_backend_classes
    from inspect_evals.bfcl.bfcl import (
        BFCL_GITHUB_COMMIT,
        DATASET_LOCATION,
        EVAL_VERSION,
        FUNC_SOURCE_LOCATION,
        GITHUB_DATA_PATH,
        GITHUB_FUNC_SOURCE_PATH,
        GITHUB_REPO_URL,
        bfcl_scorer,
        bfcl_solver,
        multi_turn_record_to_sample,
        single_turn_record_to_sample,
    )
    from inspect_evals.bfcl.data import (
        BFCLRecord,
        download_github_directory,
        load_records_by_category,
        sanitize_snapshot_id,
    )
    from inspect_evals.bfcl.utils.task_categories import (
        CATEGORIES,
        CATEGORY_ALIASES,
        FILENAME_PREFIX,
        FILENAME_SUFFIX,
        V3_CATEGORIES,
        V4_CATEGORIES,
        validate_categories,
    )

    raw = SCORED_CATEGORIES if categories is None else categories
    raw_names = [raw] if isinstance(raw, str) else list(raw)
    names: list[str] = []
    for name in raw_names:
        names.extend(STRATA.get(name) or CATEGORY_ALIASES.get(name) or [name])
    validate_categories(
        category_names=names,
        valid_categories=set(CATEGORIES) - {"rest", "format_sensitivity"},
    )

    canary = DATASET_LOCATION / f"{FILENAME_PREFIX}simple_python{FILENAME_SUFFIX}"
    if not canary.exists():
        DATASET_LOCATION.mkdir(parents=True, exist_ok=True)
        try:
            download_github_directory(
                repo_url=GITHUB_REPO_URL,
                commit=BFCL_GITHUB_COMMIT,
                sparse_path=GITHUB_DATA_PATH,
                local_dir=DATASET_LOCATION,
            )
        except Exception:
            shutil.rmtree(DATASET_LOCATION, True)
            raise
    stateful = any(c in V3_CATEGORIES | V4_CATEGORIES for c in names)
    if stateful and not FUNC_SOURCE_LOCATION.exists():
        FUNC_SOURCE_LOCATION.mkdir(parents=True, exist_ok=True)
        try:
            download_func_source(
                dest_dir=FUNC_SOURCE_LOCATION,
                repo_url=GITHUB_REPO_URL,
                commit=BFCL_GITHUB_COMMIT,
                sparse_path=GITHUB_FUNC_SOURCE_PATH,
            )
        except Exception:
            shutil.rmtree(FUNC_SOURCE_LOCATION, True)
            raise
    if stateful:
        setup_backend_classes(FUNC_SOURCE_LOCATION)

    resolved = sanitize_snapshot_id(snapshot_id) if snapshot_id else None
    records: dict[str, BFCLRecord] = {}
    for name in names:
        records.update(
            load_records_by_category(
                category_name=name, cache_dir=DATASET_LOCATION, snapshot_id=resolved
            )
        )

    samples = []
    for key in sorted(records):
        record = records[key]
        config = CATEGORIES[record.category_name]
        if config.is_multi_turn or config.is_agentic:
            samples.append(multi_turn_record_to_sample(record))
        else:
            samples.append(single_turn_record_to_sample(record))
    for sample in samples:
        stratum = next((s for s, cs in STRATA.items() if sample.metadata["category_name"] in cs), None)
        sample.metadata["stratum"] = stratum

    return Task(
        dataset=MemoryDataset(samples=samples, name="BFCL", location=str(DATASET_LOCATION)),
        solver=bfcl_solver(),
        scorer=bfcl_scorer(),
        version=EVAL_VERSION.comparability_version,
        metadata={**EVAL_VERSION.to_metadata(), "wrapper": "bench/bfcl"},
    )
