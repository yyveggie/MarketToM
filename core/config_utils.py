# -*- coding: utf-8 -*-

import copy
import os
from typing import Any, Dict, Optional, Tuple


LEGACY_MODEL_PARAM_SECTIONS = {
    "forward": "forward_inference_params",
    "action_probability": "action_probability_params",
    "backward": "backward_inference_params",
}

LEGACY_PARAM_ALIASES = {
    "action_probability": {
        "max_retries_list": "max_retries",
        "base_delay_list_seconds": "base_delay_seconds",
    },
}


def deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def get_active_provider_config(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    api_config = config.get("api", {})
    active_provider = str(api_config.get("active_llm_provider", "openai")).lower()
    provider_configs = api_config.get("providers", {})
    provider_config = provider_configs.get(active_provider)
    if provider_config is None:
        for provider_name, candidate in provider_configs.items():
            if str(provider_name).lower() == active_provider:
                provider_config = candidate
                break
    return active_provider, provider_config or {}


def resolve_api_key(provider_config: Dict[str, Any]) -> Optional[str]:
    api_key = provider_config.get("api_key")
    api_key_env = provider_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env, api_key)
    return api_key


def normalize_model_params(section: str, params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    aliases = LEGACY_PARAM_ALIASES.get(section, {})
    for key, value in params.items():
        normalized[aliases.get(key, key)] = value
    return normalized


def get_model_params(
    config: Dict[str, Any],
    section: str,
    defaults: Optional[Dict[str, Any]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = dict(defaults or {})

    if provider_config and "llm_temperature" in provider_config:
        params["llm_temperature"] = provider_config["llm_temperature"]

    legacy_section = LEGACY_MODEL_PARAM_SECTIONS.get(section)
    if legacy_section:
        legacy_params = config.get(legacy_section, {})
        if isinstance(legacy_params, dict):
            params.update(normalize_model_params(section, legacy_params))

    section_params = config.get("model_params", {}).get(section, {})
    if isinstance(section_params, dict):
        params.update(section_params)

    return params
