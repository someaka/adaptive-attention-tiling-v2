"""Utility module for loading test configurations."""

import os
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy

import yaml


def get_config_dir() -> Path:
    """Get the path to the config directory."""
    workspace_root = Path(__file__).parent.parent.parent
    return workspace_root / "configs" / "test_regimens"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary to override base values
        
    Returns:
        Merged dictionary
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
            
    return result


def load_test_config(profile: str | None = None) -> Dict[str, Any]:
    """Load test configuration based on the specified profile.
    
    Args:
        profile: The hardware profile to load ('tiny', 'standard', or 'server').
                If None, uses TEST_REGIME environment variable or falls back to 'debug'.
        
    Returns:
        Dict containing the merged test configuration
        
    Raises:
        ValueError: If the profile doesn't exist
    """
    if profile is None:
        profile = os.environ.get("TEST_REGIME", "debug")
    
    config_dir = get_config_dir()
    base_path = config_dir / "base.yaml"
    profile_path = config_dir / f"{profile}.yaml"
    
    if not profile_path.exists():
        raise ValueError(
            f"Configuration profile '{profile}' not found. "
            f"Available profiles: {[p.stem for p in config_dir.glob('*.yaml') if p.stem != 'base']}"
        )
    
    # Load base config
    with open(base_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # Load profile config
    with open(profile_path, "r") as f:
        profile_config = yaml.safe_load(f)
        
    # Merge configs with profile taking precedence
    return deep_merge(base_config, profile_config)


def get_available_profiles() -> list[str]:
    """Get list of available test configuration profiles."""
    return [p.stem for p in get_config_dir().glob("*.yaml") if p.stem != "base"] 