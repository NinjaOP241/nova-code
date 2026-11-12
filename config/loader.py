from email.mime import base
from importlib.resources import path
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir, user_data_dir
import tomli
from dotenv import load_dotenv

from config.config import Config
from utils.errors import ConfigError

import logging

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "config.toml"
AGENT_MD_FILE = "AGENT.MD"


def get_config_dir() -> Path:
    r"""
    Return the OS-specific directory where this application stores its config.

    We use platformdirs instead of hardcoding a path because configuration
    directories differ between Windows, macOS, and Linux.

    For example, we don't want to assume that every operating system stores
    application configuration under the same directory.

    On Windows, this typically resolves to:
        C:\Users\USER\AppData\Local\nova-code
    """

    return Path(user_config_dir("nova-code", appauthor=False))


def get_data_dir() -> Path:
    return Path(user_data_dir("nova-code", appauthor=False))


def get_system_config_path() -> Path:
    """
    Build the path to the user's system-wide configuration file.

    This is the global configuration shared by projects using this application.
    A project can override its settings with its own config.toml.
    """

    return get_config_dir() / CONFIG_FILE_NAME


def _parse_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and convert it into a Python dictionary."""

    try:
        with open(path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise ConfigError(
            f"Invalid TOML in {path}: {e}",
            config_file=str(path),
        ) from e
    except (OSError, IOError) as e:
        raise ConfigError(
            f"Failed to read config file {path}: {e}",
            config_file=str(path),
        ) from e


def _get_project_config(cwd: Path) -> Path | None:
    """
    Find the configuration file belonging to the current project.

    We look for this structure:

        project/
        ├── .ai-agent/
        │   └── config.toml
        ├── src/
        └── ...

    The .ai-agent directory keeps agent-specific project configuration
    separate from the project's source code.

    Return the config file path if it exists; otherwise return None.
    """

    # resolve() converts the path into an absolute path
    current = cwd.resolve()
    agent_dir = current / ".ai-agent"

    if agent_dir.is_dir():
        config_file = agent_dir / CONFIG_FILE_NAME

        if config_file.is_file():
            return config_file

    return None


def _get_agent_md_files(cwd: Path) -> str | None:
    current = cwd.resolve()

    if current.is_dir():
        agent_md_file = current / AGENT_MD_FILE

        if agent_md_file.is_file():
            return agent_md_file.read_text(encoding="utf-8")

    return None


def _merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
):
    """
    Merge two configuration dictionaries, giving priority to override.

    Think of:
        base     = system-wide configuration
        override = project-specific configuration

    If both dictionaries contain the same key:
    - If both values are dictionaries, merge them recursively.
    - Otherwise, use the value from override.

    For example:

        base = {
            "model": {
                "name": "model-a",
                "temperature": 1
            }
        }

        override = {
            "model": {
                "name": "model-b"
            }
        }

        Result = {
            "model": {
                "name": "model-b",
                "temperature": 1
            }
        }

    Lists and other non-dictionary values are replaced, not combined.
    """

    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def load_config(cwd: Path | None) -> Config:
    cwd = cwd or Path.cwd()
    load_dotenv(cwd / ".env")

    config_dict: dict[str, Any] = {}

    # ---------------------------------------------------------
    # 1. Load system-wide configuration
    # ---------------------------------------------------------
    system_config_path = get_system_config_path()

    if system_config_path.is_file():
        try:
            config_dict = _parse_toml(system_config_path)
        except ConfigError:
            logger.warning(f"Skipping invalid system config: {system_config_path}")

    # ---------------------------------------------------------
    # 2. Load project-specific configuration
    # ---------------------------------------------------------
    project_config_path = _get_project_config(cwd)

    if project_config_path:
        try:
            project_config_dict = _parse_toml(project_config_path)

            # Project configuration has higher priority than system
            # configuration. It overrides matching values while preserving
            # unrelated settings through the recursive merge.
            config_dict = _merge_dicts(config_dict, project_config_dict)

        except ConfigError:
            logger.warning(f"Skipping invalid project config: {project_config_path}")

    # ---------------------------------------------------------
    # 3. Set the working directory if it wasn't configured
    # ---------------------------------------------------------
    if "cwd" not in config_dict:
        config_dict["cwd"] = cwd

    # ---------------------------------------------------------
    # 4. Load AGENT.MD as developer instructions
    # ---------------------------------------------------------
    if "developer_instructions" not in config_dict:
        agent_md_content = _get_agent_md_files(cwd)

        if agent_md_content:
            config_dict["developer_instructions"] = agent_md_content

    # ---------------------------------------------------------
    # 5. Create and validate the final configuration object
    # ---------------------------------------------------------
    try:
        config = Config(**config_dict)
    except ConfigError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e

    return config
