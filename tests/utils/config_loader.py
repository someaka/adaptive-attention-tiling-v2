"""Utility module for loading test configurations."""

import os
from pathlib import Path
from typing import Dict, Any

import yaml


def get_config_dir() -> Path:
    """Get the path to the config directory."""
    workspace_root = Path(__file__).parent.parent.parent
    return workspace_root / "configs" / "test_regimens"


def load_test_config(profile: str | None = None) -> Dict[str, Any]:
    """Load test configuration based on the specified profile.
    
    Args:
        profile: The hardware profile to load ('tiny', 'standard', or 'server').
                If None, uses PYTEST_PROFILE environment variable or falls back to 'tiny'.
        
    Returns:
        Dict containing the test configuration
        
    Raises:
        ValueError: If the profile doesn't exist
    """
    if profile is None:
        profile = os.environ.get("PYTEST_PROFILE", "tiny")
    
    config_path = get_config_dir() / f"{profile}.yaml"
    
    if not config_path.exists():
        raise ValueError(
            f"Configuration profile '{profile}' not found. "
            f"Available profiles: {[p.stem for p in get_config_dir().glob('*.yaml')]}"
        )
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_available_profiles() -> list[str]:
    """Get list of available test configuration profiles."""
    return [p.stem for p in get_config_dir().glob("*.yaml")] 