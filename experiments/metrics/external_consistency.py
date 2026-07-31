"""Same-input cross-LLM consistency analysis of MarketToM's inferred mental states.

This analysis does NOT provide external or behavioural validation of the inferred
mental states. It asks a narrower question: when an independent evaluator LLM is
shown *only* the same five days of quote and text evidence plus a fixed rubric,
do its coarse labels agree with the labels obtained by mapping MarketToM's
open-text state descriptions into that same rubric?

Protocol (two stages, both run by the evaluator model, never by the inference
backbone):
  Stage 1 - reference labelling: the evaluator sees only the environmental state
            and the rubric, and assigns one label per (role, dimension).
  Stage 2 - description mapping: the evaluator maps MarketToM's open-text state
            descriptions into the identical label space, without ever seeing the
            Stage 1 reference labels, the prediction, or the realized return.

Agreement and Cohen's kappa are then computed per dimension, per agent role, per
(role, dimension) cell, and per instance. Every run writes a provenance record so
that the evaluator model, rubric, sampling rule, and sample size are recoverable
from the artifacts alone.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import openai


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config_utils import get_active_provider_config, resolve_api_key

ANALYSIS_NAME = "same-input cross-LLM consistency"
ANALYSIS_VERSION = "2.0"

DEFAULT_ROLES = ("Retail", "Institutional", "Arbitrageur")
DIMENSIONS = ("belief", "intention", "emotion")

# Mental-state dimension -> key used inside the inference log's agent_results.
STATE_KEYS = {"belief": "belief", "intention": "intent", "emotion": "emotion"}

# Fixed coarse label space. Reported verbatim in the manuscript.
DEFAULT_RUBRIC = {
    "belief": ("bullish", "neutral", "bearish"),
    "intention": ("buy", "hold", "sell"),
    "emotion": ("positive", "neutral", "negative"),
}

DEFAULT_SETTINGS = {
    "evaluator_provider": None,        # None -> fall back to api.active_llm_provider (and warn)
    "evaluator_model": None,           # None -> provider's llm_model_default
    "evaluator_temperature": 0.0,      # deterministic labelling; ignored on Anthropic providers
    "evaluator_max_tokens": 2048,      # Anthropic requires an explicit output cap
    "max_retries": 5,
    "retry_base_delay_seconds": 2.0,
    "sleep_between_instances": 0.5,
    "sample_size": None,               # None -> use every eligible log
    "sampling": "random",              # "random" | "first" | "all"
    "random_seed": 20260730,
    "allow_same_model": False,         # refuse a same-model run unless explicitly enabled
}

SUMMARY_FIELDS = ["scope", "key", "pairs", "agreement", "cohen_kappa"]
PAIR_FIELDS = ["instance", "role", "dimension", "reference_label", "mapped_label", "match"]


# ---------------------------------------------------------------- config utils

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path):
    return json.loads(resolve_path(config_path).read_text(encoding="utf-8"))


def get_settings(config, overrides=None):
    settings = dict(DEFAULT_SETTINGS)
    settings.update(config.get("external_consistency", {}) or {})
    for key, value in (overrides or {}).items():
        if value is not None:
            settings[key] = value
    return settings


def get_rubric(config):
    configured = (config.get("external_consistency", {}) or {}).get("rubric")
    if not configured:
        return {dim: tuple(labels) for dim, labels in DEFAULT_RUBRIC.items()}
    rubric = {}
    for dimension in DIMENSIONS:
        labels = configured.get(dimension) or DEFAULT_RUBRIC[dimension]
        rubric[dimension] = tuple(str(label).strip().lower() for label in labels)
    return rubric


def get_roles(config):
    roles = (config.get("agent_params", {}) or {}).get("agent_roles")
    return tuple(roles) if roles else DEFAULT_ROLES


def get_provider(config, provider_name):
    providers = (config.get("api", {}) or {}).get("providers", {}) or {}
    if provider_name in providers:
        return providers[provider_name]
    for name, candidate in providers.items():
        if str(name).lower() == str(provider_name).lower():
            return candidate
    raise ValueError(
        f"Evaluator provider '{provider_name}' is not defined under api.providers in the config."
    )


def resolve_evaluator(config, settings, log_backbones=None):
    """Pick the evaluator provider/model and check it differs from the inference backbone.

    The inference backbone is taken from the run metadata recorded inside the logs
    whenever available, because the config may have changed since the logs were
    produced. It falls back to the config's active provider otherwise.
    """
    inference_provider_name, inference_provider = get_active_provider_config(config)
    config_inference_model = inference_provider.get("llm_model_default")

    log_backbones = sorted(log_backbones or [])
    if log_backbones:
        inference_models = log_backbones
        backbone_source = "log run_metadata"
    else:
        inference_models = [config_inference_model] if config_inference_model else []
        backbone_source = "config api.active_llm_provider (logs carry no run_metadata)"

    evaluator_provider_name = settings.get("evaluator_provider") or inference_provider_name
    evaluator_provider = get_provider(config, evaluator_provider_name)
    evaluator_model = settings.get("evaluator_model") or evaluator_provider.get("llm_model_default")
    if not evaluator_model:
        raise ValueError(
            f"No evaluator model resolved for provider '{evaluator_provider_name}'. "
            "Set external_consistency.evaluator_model or the provider's llm_model_default."
        )

    collision = [model for model in inference_models if str(model) == str(evaluator_model)]
    if collision and not settings.get("allow_same_model"):
        raise ValueError(
            f"The evaluator model ('{evaluator_model}') matches the inference backbone recorded in "
            f"{backbone_source}, so this run would measure same-model self-agreement rather than "
            "cross-LLM consistency. Set external_consistency.evaluator_provider to a different "
            "provider, or pass --allow-same-model to override deliberately."
        )
    if collision:
        print(
            "\033[1;33m[warn] Evaluator and inference backbone are the same model; "
            "results measure same-model self-agreement, not cross-LLM consistency.\033[0m"
        )

    return {
        "inference_provider": inference_provider_name,
        "inference_models": inference_models,
        "inference_backbone_source": backbone_source,
        "evaluator_provider": evaluator_provider_name,
        "evaluator_model": evaluator_model,
        "evaluator_config": evaluator_provider,
        "same_model": bool(collision),
    }


# ------------------------------------------------------------- evaluator clients
#
# Two request surfaces are supported. OpenAI-compatible providers (OpenAI itself,
# Moonshot, xAI, and OpenAI-compatible Qwen endpoints) go through the openai SDK.
# Anthropic models use the Messages API via the official anthropic SDK, because
# the two surfaces differ in ways that matter here: Claude Opus 4.7 rejects
# `temperature` outright, and JSON output is not requested through OpenAI's
# `response_format` parameter.

def extract_json_object(text):
    """Parse a JSON object from a model response, tolerating surrounding prose."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON object found in response", text or "", 0)


class OpenAICompatibleEvaluator:
    """Chat-completions surface: OpenAI, Moonshot, xAI, OpenAI-compatible Qwen."""

    surface = "openai_compatible"

    def __init__(self, provider_config, model, settings):
        api_key = resolve_api_key(provider_config)
        if not api_key:
            raise ValueError(
                "API key missing for the evaluator provider; set the environment variable named "
                f"by api_key_env ({provider_config.get('api_key_env')})."
            )
        client_kwargs = {"api_key": api_key}
        if provider_config.get("base_url"):
            client_kwargs["base_url"] = provider_config["base_url"]
        if provider_config.get("timeout"):
            client_kwargs["timeout"] = float(provider_config["timeout"])
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.temperature = float(settings["evaluator_temperature"])
        self.extra_body = provider_config.get("extra_body")

    def decoding_record(self):
        return {"surface": self.surface, "temperature": self.temperature,
                "json_mode": "response_format=json_object"}

    def complete_json(self, prompt):
        request_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.extra_body:
            request_kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**request_kwargs)
        return extract_json_object(response.choices[0].message.content)


class AnthropicEvaluator:
    """Messages API surface for Claude evaluators.

    Notes specific to Claude Opus 4.7 and later:
      * `temperature` was removed and returns a 400 -- it is never sent.
      * Omitting `thinking` runs the model without extended thinking, which is
        what this coarse rubric-labelling task wants.
      * JSON is requested in the prompt and parsed here, rather than through
        OpenAI's `response_format` parameter, which does not exist on this API.
    """

    surface = "anthropic_messages"

    def __init__(self, provider_config, model, settings):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The anthropic SDK is required for a Claude evaluator. Install it with "
                "`pip install anthropic`."
            ) from exc
        api_key = resolve_api_key(provider_config)
        if not api_key:
            raise ValueError(
                "API key missing for the evaluator provider; set the environment variable named "
                f"by api_key_env ({provider_config.get('api_key_env')})."
            )
        client_kwargs = {"api_key": api_key}
        if provider_config.get("base_url"):
            client_kwargs["base_url"] = provider_config["base_url"]
        if provider_config.get("timeout"):
            client_kwargs["timeout"] = float(provider_config["timeout"])
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self.max_tokens = int(settings["evaluator_max_tokens"])

    def decoding_record(self):
        return {
            "surface": self.surface,
            "temperature": "not sent (removed on Claude Opus 4.7 and later)",
            "thinking": "omitted (runs without extended thinking)",
            "max_tokens": self.max_tokens,
            "json_mode": "requested in prompt, parsed client-side",
        }

    def complete_json(self, prompt):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return extract_json_object(text)


def build_evaluator_client(provider_config, model, settings):
    surface = str(provider_config.get("type", "openai_compatible")).lower()
    if surface in ("anthropic", "anthropic_messages"):
        return AnthropicEvaluator(provider_config, model, settings)
    return OpenAICompatibleEvaluator(provider_config, model, settings)


def call_json(client, prompt, max_retries, base_delay):
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.complete_json(prompt), None
        except Exception as exc:  # noqa: BLE001 - surfaced in the provenance record
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(base_delay * (attempt + 1))
    return {}, last_error


def rubric_block(rubric):
    return "\n".join(f"- {dim}: {' | '.join(rubric[dim])}" for dim in DIMENSIONS)


def evaluator_prompt(env_state, rubric, roles):
    """Stage 1: reference labels from the market evidence alone."""
    return (
        "You are an independent market analyst. Using only the five trading days of quote and text "
        "information below and the fixed rubric, assign one coarse label for the belief, intention, and "
        "emotion of each market participant role. Use only the allowed labels.\n\n"
        f"Allowed labels:\n{rubric_block(rubric)}\n\n"
        f"Roles: {', '.join(roles)}\n\n"
        f"Market information:\n{env_state}\n\n"
        "Respond with a JSON object mapping each role to an object with keys belief, intention, emotion."
    )


def mapper_prompt(agent_results, rubric, roles):
    """Stage 2: map open-text state descriptions into the identical label space."""
    lines = []
    for role in roles:
        states = agent_results.get(role, {})
        for dimension in DIMENSIONS:
            text = states.get(STATE_KEYS[dimension], "")
            lines.append(f"{role} {dimension}: {text}")
    descriptions = "\n".join(lines)
    return (
        "Map each open-text mental-state description to the single closest coarse label, using only the "
        "allowed labels. Do not add information beyond the description itself.\n\n"
        f"Allowed labels:\n{rubric_block(rubric)}\n\n"
        f"Descriptions:\n{descriptions}\n\n"
        "Respond with a JSON object mapping each role to an object with keys belief, intention, emotion."
    )


# ------------------------------------------------------------------ label utils

def normalize_label(dimension, value, rubric):
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    allowed = rubric[dimension]
    if candidate in allowed:
        return candidate
    for label in allowed:
        if label in candidate:
            return label
    return None


def extract_labels(payload, rubric, roles):
    labels = {}
    for role in roles:
        role_payload = payload.get(role, {}) if isinstance(payload, dict) else {}
        if not isinstance(role_payload, dict):
            role_payload = {}
        labels[role] = {
            dimension: normalize_label(dimension, role_payload.get(dimension), rubric)
            for dimension in DIMENSIONS
        }
    return labels


def cohen_kappa(pairs):
    """Return (observed agreement, Cohen's kappa) for a list of (reference, mapped) pairs."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    agree = sum(1 for reference, mapped in pairs if reference == mapped)
    po = agree / n
    classes = set()
    reference_counts = defaultdict(int)
    mapped_counts = defaultdict(int)
    for reference, mapped in pairs:
        reference_counts[reference] += 1
        mapped_counts[mapped] += 1
        classes.add(reference)
        classes.add(mapped)
    pe = sum((reference_counts[label] / n) * (mapped_counts[label] / n) for label in classes)
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0
    return po, kappa


# ------------------------------------------------------------------ log loading

def load_eligible_logs(logs_dir, roles):
    """Return (eligible, report). Eligible entries carry role-resolved mental states."""
    directory = resolve_path(logs_dir)
    paths = sorted(directory.glob("inference_*.json"))
    eligible = []
    backbones = set()
    report = {
        "logs_dir": str(directory),
        "files_found": len(paths),
        "skipped_unreadable": 0,
        "skipped_missing_env_state": 0,
        "skipped_legacy_no_agent_results": 0,
        "skipped_incomplete_roles": 0,
        "logs_without_run_metadata": 0,
    }
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["skipped_unreadable"] += 1
            continue
        env_state = data.get("environmental_state")
        if not env_state:
            report["skipped_missing_env_state"] += 1
            continue
        agent_results = data.get("agent_results")
        if not isinstance(agent_results, dict) or not agent_results:
            # Logs written before role-resolved traces existed store a flat
            # "mental_states" dict, which cannot be attributed to an agent role.
            report["skipped_legacy_no_agent_results"] += 1
            continue
        if any(role not in agent_results for role in roles):
            report["skipped_incomplete_roles"] += 1
            continue
        run_metadata = data.get("run_metadata") or {}
        backbone = run_metadata.get("llm_model")
        if backbone:
            backbones.add(backbone)
        else:
            report["logs_without_run_metadata"] += 1
        eligible.append((path, env_state, agent_results))
    report["eligible"] = len(eligible)
    report["inference_backbones_in_logs"] = sorted(backbones)
    return eligible, report


def select_sample(eligible, settings):
    """Apply the configured sampling rule and return (sample, sampling_record)."""
    mode = str(settings.get("sampling", "random")).lower()
    size = settings.get("sample_size")
    seed = settings.get("random_seed")

    if mode == "all" or size is None or size >= len(eligible):
        sample = list(eligible)
        applied = "all"
        effective_seed = None
    elif mode == "first":
        sample = list(eligible)[:size]
        applied = "first"
        effective_seed = None
    elif mode == "random":
        rng = random.Random(seed)
        sample = rng.sample(list(eligible), size)
        sample.sort(key=lambda item: item[0].name)
        applied = "random"
        effective_seed = seed
    else:
        raise ValueError(f"Unknown sampling mode '{mode}'. Use 'random', 'first', or 'all'.")

    record = {
        "requested_mode": mode,
        "applied_mode": applied,
        "requested_sample_size": size,
        "realized_sample_size": len(sample),
        "random_seed": effective_seed,
        "ordering": "sorted by log filename",
        "instances": [path.name for path, _, _ in sample],
    }
    return sample, record


# -------------------------------------------------------------------- analysis

def evaluate(sample, evaluator, settings, rubric, roles, client=None):
    if client is None:
        client = build_evaluator_client(
            evaluator["evaluator_config"], evaluator["evaluator_model"], settings
        )
    max_retries = int(settings["max_retries"])
    base_delay = float(settings["retry_base_delay_seconds"])
    sleep_seconds = float(settings["sleep_between_instances"])

    pair_records = []
    failures = {"reference_call": 0, "mapping_call": 0, "unmappable_labels": 0}

    total = len(sample)
    for index, (path, env_state, agent_results) in enumerate(sample, start=1):
        print(f"  [{index}/{total}] {path.name}")

        reference_payload, reference_error = call_json(
            client, evaluator_prompt(env_state, rubric, roles), max_retries, base_delay,
        )
        if reference_error:
            failures["reference_call"] += 1

        mapped_payload, mapping_error = call_json(
            client, mapper_prompt(agent_results, rubric, roles), max_retries, base_delay,
        )
        if mapping_error:
            failures["mapping_call"] += 1

        reference = extract_labels(reference_payload, rubric, roles)
        mapped = extract_labels(mapped_payload, rubric, roles)

        for role in roles:
            for dimension in DIMENSIONS:
                reference_label = reference[role][dimension]
                mapped_label = mapped[role][dimension]
                if not reference_label or not mapped_label:
                    failures["unmappable_labels"] += 1
                    continue
                pair_records.append({
                    "instance": path.name,
                    "role": role,
                    "dimension": dimension,
                    "reference_label": reference_label,
                    "mapped_label": mapped_label,
                    "match": int(reference_label == mapped_label),
                })

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return pair_records, failures


def build_summary(pair_records, roles):
    """Summarize by dimension, by role, by (role, dimension), and by instance."""
    by_dimension = defaultdict(list)
    by_role = defaultdict(list)
    by_cell = defaultdict(list)
    by_instance = defaultdict(list)

    for record in pair_records:
        pair = (record["reference_label"], record["mapped_label"])
        by_dimension[record["dimension"]].append(pair)
        by_role[record["role"]].append(pair)
        by_cell[(record["role"], record["dimension"])].append(pair)
        by_instance[record["instance"]].append(pair)

    rows = []

    for dimension in DIMENSIONS:
        pairs = by_dimension[dimension]
        agreement, kappa = cohen_kappa(pairs)
        rows.append({"scope": "dimension", "key": dimension,
                     "pairs": len(pairs), "agreement": agreement, "cohen_kappa": kappa})

    dimension_rows = [row for row in rows if row["pairs"]]
    if dimension_rows:
        rows.append({
            "scope": "dimension", "key": "average",
            "pairs": sum(row["pairs"] for row in dimension_rows),
            "agreement": sum(row["agreement"] for row in dimension_rows) / len(dimension_rows),
            "cohen_kappa": sum(row["cohen_kappa"] for row in dimension_rows) / len(dimension_rows),
        })

    for role in roles:
        pairs = by_role[role]
        agreement, kappa = cohen_kappa(pairs)
        rows.append({"scope": "role", "key": role,
                     "pairs": len(pairs), "agreement": agreement, "cohen_kappa": kappa})

    for role in roles:
        for dimension in DIMENSIONS:
            pairs = by_cell[(role, dimension)]
            agreement, kappa = cohen_kappa(pairs)
            rows.append({"scope": "role_dimension", "key": f"{role}/{dimension}",
                         "pairs": len(pairs), "agreement": agreement, "cohen_kappa": kappa})

    all_pairs = [(r["reference_label"], r["mapped_label"]) for r in pair_records]
    agreement, kappa = cohen_kappa(all_pairs)
    rows.append({"scope": "overall", "key": "pooled",
                 "pairs": len(all_pairs), "agreement": agreement, "cohen_kappa": kappa})

    if by_instance:
        per_instance = [sum(1 for ref, mapped in pairs if ref == mapped) / len(pairs)
                        for pairs in by_instance.values() if pairs]
        rows.append({"scope": "instance", "key": "mean_within_instance_agreement",
                     "pairs": len(per_instance),
                     "agreement": sum(per_instance) / len(per_instance) if per_instance else 0.0,
                     "cohen_kappa": float("nan")})

    return rows


# -------------------------------------------------------------------- artifacts

def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scope": row["scope"],
                "key": row["key"],
                "pairs": row["pairs"],
                "agreement": f"{row['agreement']:.4f}",
                "cohen_kappa": "" if row["cohen_kappa"] != row["cohen_kappa"] else f"{row['cohen_kappa']:.4f}",
            })


def write_pairs_csv(path, pair_records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pair_records)


def write_tex(path, rows, metadata):
    """Manuscript-ready table: one row per dimension, with the label-pair count."""
    dimension_rows = [row for row in rows if row["scope"] == "dimension"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by experiments/metrics/external_consistency.py",
        f"% evaluator: {metadata['evaluator']['evaluator_model']} "
        f"(provider: {metadata['evaluator']['evaluator_provider']}, "
        f"temperature: {metadata['settings']['evaluator_temperature']})",
        f"% instances: {metadata['sampling']['realized_sample_size']} "
        f"({metadata['sampling']['applied_mode']} sampling, seed: {metadata['sampling']['random_seed']})",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Same-input cross-LLM consistency of inferred mental states.}",
        "\\label{tab:external_consistency}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lccc}",
        "\\toprule",
        "\\textbf{Mental-state dimension} & \\textbf{Label pairs} & "
        "\\textbf{Agreement with reference labels} & \\textbf{Cohen's $\\kappa$} \\\\",
        "\\midrule",
    ]
    for row in dimension_rows:
        label = str(row["key"]).capitalize()
        if row["key"] == "average":
            lines.append("\\midrule")
        lines.append(
            f"{label} & {row['pairs']} & {row['agreement'] * 100:.2f}\\% & {row['cohen_kappa']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Same-input cross-LLM consistency analysis of inferred mental states"
    )
    parser.add_argument("--logs-dir", default="runtime_storage/inference_logs")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs/metrics/external_consistency")
    parser.add_argument("--evaluator-provider", default=None,
                        help="Provider key for the evaluator LLM; must differ from the inference backbone")
    parser.add_argument("--evaluator-model", default=None,
                        help="Explicit evaluator model id; defaults to the provider's llm_model_default")
    parser.add_argument("--evaluator-temperature", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of inference logs to evaluate; omit to use every eligible log")
    parser.add_argument("--sampling", choices=["random", "first", "all"], default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--allow-same-model", action="store_true", default=None,
                        help="Permit an evaluator identical to the inference backbone (not cross-LLM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve logs, sampling, and evaluator without issuing any API call")
    # Backward-compatible alias for the previous interface.
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Deprecated alias for --sample-size with --sampling first")
    parser.add_argument("--sleep", type=float, default=None,
                        help="Seconds to sleep between instances")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    overrides = {
        "evaluator_provider": args.evaluator_provider,
        "evaluator_model": args.evaluator_model,
        "evaluator_temperature": args.evaluator_temperature,
        "sample_size": args.sample_size,
        "sampling": args.sampling,
        "random_seed": args.random_seed,
        "allow_same_model": args.allow_same_model,
        "sleep_between_instances": args.sleep,
    }
    if args.max_instances is not None and args.sample_size is None:
        overrides["sample_size"] = args.max_instances
        if args.sampling is None:
            overrides["sampling"] = "first"

    settings = get_settings(config, overrides)
    rubric = get_rubric(config)
    roles = get_roles(config)

    eligible, log_report = load_eligible_logs(args.logs_dir, roles)
    evaluator = resolve_evaluator(config, settings, log_report["inference_backbones_in_logs"])

    print(f"\nAnalysis      : {ANALYSIS_NAME} (v{ANALYSIS_VERSION})")
    print(f"Logs dir      : {log_report['logs_dir']}")
    print(f"Files found   : {log_report['files_found']}")
    print(f"Eligible      : {log_report['eligible']}")
    if log_report["skipped_legacy_no_agent_results"]:
        print(f"\033[1;33m[warn] {log_report['skipped_legacy_no_agent_results']} log(s) lack role-resolved "
              f"'agent_results' and were skipped. These predate role-resolved traces and cannot be "
              f"attributed to an agent role.\033[0m")
    if not eligible:
        raise SystemExit(
            "No eligible inference logs. Run the forward inference pipeline first so that logs "
            "contain role-resolved 'agent_results', then re-run this analysis."
        )

    sample, sampling_record = select_sample(eligible, settings)
    print(f"Evaluator     : {evaluator['evaluator_model']} (provider: {evaluator['evaluator_provider']})")
    print(f"Inference LLM : {', '.join(map(str, evaluator['inference_models'])) or 'unknown'} "
          f"(from {evaluator['inference_backbone_source']})")
    print(f"Cross-LLM     : {'no (same model)' if evaluator['same_model'] else 'yes'}")
    print(f"Sampling      : {sampling_record['applied_mode']}, n={sampling_record['realized_sample_size']}, "
          f"seed={sampling_record['random_seed']}")

    client = build_evaluator_client(
        evaluator["evaluator_config"], evaluator["evaluator_model"], settings
    )
    decoding = client.decoding_record()
    print(f"Request surface: {decoding['surface']}")
    print(f"Decoding      : {', '.join(f'{k}={v}' for k, v in decoding.items() if k != 'surface')}")
    print(f"Planned calls : {2 * len(sample)}")

    metadata = {
        "analysis": ANALYSIS_NAME,
        "version": ANALYSIS_VERSION,
        "started_at": datetime.now().isoformat(),
        "config_path": str(resolve_path(args.config)),
        "evaluator": {key: value for key, value in evaluator.items() if key != "evaluator_config"},
        "decoding": decoding,
        "settings": {key: value for key, value in settings.items()},
        "rubric": {dim: list(labels) for dim, labels in rubric.items()},
        "roles": list(roles),
        "log_report": log_report,
        "sampling": sampling_record,
        "protocol": {
            "stage_1": "evaluator labels the environmental state alone, using the rubric",
            "stage_2": "evaluator maps MarketToM's open-text descriptions into the same label space",
            "evaluator_blind_to": ["MarketToM predicted action", "realized label", "stage-1 reference labels"],
            "unmappable_pairs": "dropped, never imputed",
            "statistic": "observed agreement and Cohen's kappa",
        },
    }

    output_dir = resolve_path(args.output_dir)

    if args.dry_run:
        metadata["dry_run"] = True
        write_metadata(output_dir / "run_metadata.json", metadata)
        print(f"\nDry run complete; no API calls issued. Provenance written to: {output_dir}")
        return

    print("\nEvaluating...")
    pair_records, failures = evaluate(sample, evaluator, settings, rubric, roles, client=client)
    summary = build_summary(pair_records, roles)

    metadata["finished_at"] = datetime.now().isoformat()
    metadata["failures"] = failures
    metadata["total_label_pairs"] = len(pair_records)

    write_summary_csv(output_dir / "external_consistency.csv", summary)
    write_pairs_csv(output_dir / "pairwise_labels.csv", pair_records)
    write_tex(output_dir / "external_consistency.tex", summary, metadata)
    write_metadata(output_dir / "run_metadata.json", metadata)

    print(f"\nWritten to: {output_dir}")
    print(f"Total label pairs: {len(pair_records)}")
    if any(failures.values()):
        print(f"\033[1;33m[warn] failures: {failures}\033[0m")
    for row in summary:
        if row["scope"] in ("dimension", "overall"):
            kappa = "n/a" if row["cohen_kappa"] != row["cohen_kappa"] else f"{row['cohen_kappa']:.4f}"
            print(f"  {row['scope']:<10} {row['key']:<12} pairs={row['pairs']:<6} "
                  f"agreement={row['agreement']:.4f} kappa={kappa}")


if __name__ == "__main__":
    main()
