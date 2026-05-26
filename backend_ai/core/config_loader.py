import os
import logging
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_agents_config() -> Dict[str, Any]:
    """
    Loads the agents_config.yaml file from the backend_ai directory.
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "agents_config.yaml")
    if not os.path.exists(config_path):
        # Fallback for different execution contexts
        config_path = os.path.join(os.getcwd(), "backend_ai", "agents_config.yaml")
        
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading agents_config.yaml: {e}")
        return {}

# Global config instance
AGENTS_CONFIG = load_agents_config()
