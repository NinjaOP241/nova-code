from datetime import datetime
import json
import uuid

from agent.types import AgentName
from client.llm_client import LLMClient
from config.config import Config
from config.loader import get_data_dir
from context.manager import ContextManager
from tools.registry import create_default_registry


class Session:
    """
    Holds the components and runtime state required for one agent session.

    It brings together the LLM client, tool registry, context manager,
    approval manager, MCP manager, and other components needed by the
    agent to process user requests.

    It also tracks session-specific information such as the session ID,
    creation time, last update time, and number of turns.

    This keeps all components and state associated with a session
    organized in one place.
    """

    def __init__(
        self, config: Config, agent_name: AgentName = AgentName.MAIN_AGENT
    ) -> None:
        # Keep the configuration used throughout this session.
        # Other components receive this same configuration so they
        # follow the same model, working directory, and other settings.
        self.config = config

        # Client responsible for communicating with the LLM.
        # We create it once and reuse it throughout the session.
        self.client = LLMClient(config)

        # Create the registry containing the tools the agent can use.
        self.tool_registry = create_default_registry(config)

        self.context_manager = ContextManager(
            config,
            self._load_memory(),
            self.tool_registry.get_tools(),
            agent_name,
        )

        # Generate a unique identifier for this session.
        self.session_id = str(uuid.uuid4())

        # Record when this session was created.
        self.created_at = datetime.now()

        # Initially, the last update time is the creation time.
        # This will be refreshed when the session progresses.
        self.updated_at = self.created_at

        # Tracks the total number of agentic loop iterations across this session.
        # Incremented each time the Agent starts a new iteration and used for
        # reporting session statistics.
        self._turn_count = 0

    def _load_memory(self) -> str | None:
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        memory_file_path = data_dir / "user_memory.json"

        if not memory_file_path.exists():
            return None

        try:
            content = memory_file_path.read_text(encoding="utf-8")
            data = json.loads(content)
            entries = data.get("entries")

            if not entries:
                return None

            lines = ["User preferences and notes:"]
            for key, value in entries.items():
                lines.append(f"- {key}: {value}")

            return "\n".join(lines)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            return None

    def increment_turn(self) -> int:
        """
        Record that the agent has completed another loop iteration.

        The Agent calls this during its agentic loop.
        """

        self._turn_count += 1
        self.updated_at = datetime.now()
        return self._turn_count
