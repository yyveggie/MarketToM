import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import openai


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.forward_inference as forward_module
import core.calculate_action_prob as action_module
from core.config_utils import get_active_provider_config, get_model_params, resolve_api_key
from core.forward_inference import MentalStateInference, DataLogger
from core.calculate_action_prob import ActionProbabilityCalculator

DEFAULT_ROLES = ["Retail", "Institutional", "Arbitrageur"]
BUBBLE_VIGNETTE = (
    "Market State Description:\n"
    "1. Price Conditions:\n"
    "The asset currently trades at $100, far above its estimated fundamental value of about $20. "
    "The price has risen steeply over recent sessions on accelerating volume near the top of an extended run-up.\n"
    "2. Social Media Tweets (recent):\n"
    "- Retail enthusiasm is extremely high; many posts express strong conviction the price will keep rising and urge others to buy now.\n"
    "- Momentum-chasing dominates the discussion, with little attention to valuation.\n"
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path):
    return json.loads(resolve_path(config_path).read_text(encoding="utf-8"))


def build_client(config):
    active_provider, provider = get_active_provider_config(config)
    api_key = resolve_api_key(provider)
    if not api_key:
        raise ValueError(f"API key missing for provider '{active_provider}'")
    client_kwargs = {"api_key": api_key}
    if provider.get("base_url"):
        client_kwargs["base_url"] = provider["base_url"]
    if provider.get("timeout"):
        client_kwargs["timeout"] = float(provider["timeout"])
    client = openai.OpenAI(**client_kwargs)
    model_params = get_model_params(
        config,
        "case_study",
        {"llm_temperature": 0.6},
        provider_config=provider,
    )
    return (
        client,
        provider.get("llm_model_default", "gpt-4o"),
        model_params.get("llm_temperature", 0.6),
        provider.get("extra_body"),
    )


def apply_rate_limit(config, override):
    rate_config = dict(config.get("api_rate_limit", {}))
    if override is not None:
        rate_config = {"min_request_interval": override, "default_cooldown": override}
    for module in (forward_module, action_module):
        if "min_request_interval" in rate_config:
            module.MIN_REQUEST_INTERVAL = float(rate_config["min_request_interval"])
        if "default_cooldown" in rate_config:
            module.DEFAULT_COOLDOWN = float(rate_config["default_cooldown"])
        if "max_jitter" in rate_config:
            module.MAX_JITTER = float(rate_config["max_jitter"])


def template_paths(config):
    templates = config.get("templates", {})
    forward = templates.get("forward_inference", "./prompt_templates/forward_prompt_template.xml")
    action = templates.get("expert_action_probability", "./prompt_templates/expert_action_prob_template.xml")
    return str(resolve_path(forward)), str(resolve_path(action))


def run_order(env_state, tom_order, client, model, temperature, extra_body, roles, forward_template, action_template, logs_dir):
    logger = DataLogger(log_dir_abs_path=str(logs_dir))
    inferencer = MentalStateInference(
        cep=None,
        logger=logger,
        llm_client=client,
        llm_model=model,
        forward_template_abs_path=forward_template,
        llm_temperature=temperature,
        agent_roles=roles,
        tom_order=tom_order,
        cep_enabled=False,
        llm_extra_body=extra_body,
    )
    agent_results, filename = inferencer.forward_inference(env_state)
    calculator = ActionProbabilityCalculator(
        cep=None,
        llm_client=client,
        llm_model=model,
        inference_logs_abs_path=str(logs_dir),
        action_template_abs_path=action_template,
        agent_roles=roles,
        llm_temperature=temperature,
        llm_extra_body=extra_body,
    )
    result = calculator.calculate_probability_from_file(filename)
    return agent_results, result


def build_report(env_state, roles, orders):
    report = {"environmental_state": env_state, "orders": {}}
    for tom_order, (agent_results, result) in orders.items():
        order_block = {"aggregated_p_up": result.probability, "agents": {}}
        for role in roles:
            states = agent_results.get(role, {})
            prediction = result.agent_predictions.get(role, {})
            order_block["agents"][role] = {
                "belief": states.get("belief", ""),
                "intent": states.get("intent", ""),
                "emotion": states.get("emotion", ""),
                "predicted_action": prediction.get("predicted_action", ""),
            }
        report["orders"][f"order_{tom_order}"] = order_block
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qualitative mechanism case study (first- vs second-order ToM)")
    parser.add_argument("--case", choices=["bubble", "custom"], default="bubble")
    parser.add_argument("--env-state-file", default=None)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs/case_study")
    parser.add_argument("--roles", nargs="+", default=DEFAULT_ROLES)
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--rate-limit", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.case == "custom":
        if not args.env_state_file:
            raise ValueError("custom case requires --env-state-file")
        env_state = resolve_path(args.env_state_file).read_text(encoding="utf-8")
    else:
        env_state = BUBBLE_VIGNETTE

    config = load_config(args.config)
    apply_rate_limit(config, args.rate_limit)
    client, model, temperature, extra_body = build_client(config)
    forward_template, action_template = template_paths(config)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orders = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for tom_order in args.orders:
            orders[tom_order] = run_order(
                env_state, tom_order, client, model, temperature, extra_body,
                list(args.roles), forward_template, action_template, tmp_dir,
            )

    report = build_report(env_state, list(args.roles), orders)
    (output_dir / f"case_{args.case}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Case study written to: {output_dir}")
    for order_key, block in report["orders"].items():
        actions = {role: block["agents"][role]["predicted_action"] for role in args.roles}
        print(f"{order_key}: P(up)={block['aggregated_p_up']:.3f} actions={actions}")


if __name__ == "__main__":
    main()
