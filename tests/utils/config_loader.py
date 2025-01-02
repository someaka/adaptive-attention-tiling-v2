"""Utility module for loading test configurations."""

import os
from pathlib import Path
from typing import Dict, Any

import yaml


def get_config_dir() -> Path:
    """Get the path to the config directory."""
    workspace_root = Path(__file__).parent.parent.parent
    return workspace_root / "configs" / "test_regimens"


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configurations.
    
    Args:
        base: Base configuration
        override: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result


def convert_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string values to appropriate types.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with converted values
    """
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = convert_values(value)
        elif isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in value.lower():
                    result[key] = float(value)
                else:
                    result[key] = value
            except ValueError:
                result[key] = value
        else:
            result[key] = value
            
    return result


def load_test_config(profile: str | None = None) -> Dict[str, Any]:
    """Load test configuration based on the specified profile.
    
    Args:
        profile: The hardware profile to load ('tiny', 'standard', or 'server').
                If None, uses PYTEST_PROFILE environment variable or falls back to 'debug'.
        
    Returns:
        Dict containing the test configuration
        
    Raises:
        ValueError: If the profile doesn't exist
    """
    if profile is None:
        profile = os.environ.get("PYTEST_PROFILE", "debug")
    
    config_dir = get_config_dir()
    base_path = config_dir / "base.yaml"
    profile_path = config_dir / f"{profile}.yaml"
    
    if not profile_path.exists():
        raise ValueError(
            f"Configuration profile '{profile}' not found. "
            f"Available profiles: {[p.stem for p in config_dir.glob('*.yaml')]}"
        )
    
    # Load base configuration
    with open(base_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # Load profile configuration
    with open(profile_path, "r") as f:
        profile_config = yaml.safe_load(f)
        
    # Merge configurations
    merged_config = merge_configs(base_config, profile_config)
    
    # Convert values
    return convert_values(merged_config)


def get_available_profiles() -> list[str]:
    """Get list of available test configuration profiles."""
    return [p.stem for p in get_config_dir().glob("*.yaml")] 