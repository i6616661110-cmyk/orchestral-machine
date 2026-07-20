"""Orchestral Machine — Central Configuration Module.

All parameters, model mappings, role classifications, timeouts, and execution
limits are enforced centrally from this single module. Every other source file
imports its constants from here.
"""

import os

# ---------------------------------------------------------------------------
# OpenRouter base URL
# ---------------------------------------------------------------------------
BASE_URL: str = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Generation parameters — deterministic by default
# ---------------------------------------------------------------------------
GEN_CONFIG: dict[str, float] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# ---------------------------------------------------------------------------
# Model mapping — OpenRouter model IDs per role
# Controller logic: primary → RESERVIST → HIR (Human Intervention Required)
# ---------------------------------------------------------------------------
MODEL_MAPPING: dict[str, str] = {
    "ARCHITECT": "z-ai/glm-4.7",  # $0.4-$2.................200k
    "CODER": "deepseek/deepseek-v3.2",  # $0.3-$0.4..164k
    "REVIEWER": "stepfun/step-3.5-flash:free",  # $0........256k
    "TESTER": "openai/gpt-oss-120b:exacto",  # $0.04-0.2....131k
    "VERIFIER": "minimax/minimax-m2.5",  # $0.3-$1.2........197k
    "VALIDATOR": "minimax/minimax-m2.5",  # $0.3-$1.2.......197k
    "CORRECTOR_C1": "qwen/qwen3-coder-next",  # $0,12-$0,75.262k
    "CORRECTOR_C2": "deepseek/deepseek-v3.2",  # $0.3-$0.4..164k
    "ARCHIVIST_A1": "x-ai/grok-4.1-fast",  # $0.2-$0.5......2000k
    "ARCHIVIST_A2": "stepfun/step-3.5-flash:free",  # $0....256k
    "RESERVIST": "qwen/qwen3-coder-next",  # $0,12-$0,75....262k
    # "moonshotai/kimi-k2-thinking", # $0.5-$2.5..131k
    # "z-ai/glm-4.7",  # $0.4-$2.................200k
    # "openai/gpt-5-mini", # $0.25-$2................400k
    # "minimax/minimax-m2.5",  # $0.3-$1.2........197k
    # "qwen/qwen3.5-plus-02-15", # $0.4-$2.4(7.2)
    # "qwen/qwen3.5-397b-a17b", # $0.4-$3.6
    # "qwen/qwen3-coder-next",  # $0,12-$0,75.
    # "kwaipilot/kat-coder-pro", # $0.2-$0.83
    # "deepseek/deepseek-v3.2",  # $0.3-$0.4.........164k
    # "openai/gpt-oss-120b:exacto",  # $0.04-0.2....131k
    # "stepfun/step-3.5-flash", # $0.1-$0.3
    # "arcee-ai/trinity-large-preview:free", 131k
    # "upstage/solar-pro-3:free"
}

# ---------------------------------------------------------------------------
# Role classification — Constitutional enforcement
# ---------------------------------------------------------------------------
ROLE_CLASSIFICATION: dict[str, list[str]] = {
    "decision_authority": ["verifier", "validator", "architect"],
    "executor": ["coder", "corrector", "reviewer", "tester", "archivist"],
}

ROLE_EXCLUSIVE_KEYS: dict[str, list[str]] = {
    "architect": ["plan", "execution_strategy"],
    "coder": [],
    "corrector": ["applied_fixes"],
    "verifier": ["verifier_feedback"],
    "reviewer": ["review_results", "issues"],
    "tester": ["test_results"],
    "archivist": ["archived_summary_ref", "meta_summary"],
    "validator": [],
}

PROTECTED_APPROVAL_KEYS: list[str] = [
    "approved_summary",
    "approved_meta",
    "approved_hard_reset",
]

# ---------------------------------------------------------------------------
# Execution guarantees — timeouts and retries
# ---------------------------------------------------------------------------
NODE_TIMEOUT_DEFAULT: int = 1500  # seconds (25 minutes)
NODE_TIMEOUT_LONG: int = 1800  # seconds (30 minutes) (ARCHITECT, CODER, ARCHIVIST_A2)
NODE_HARD_TIMEOUT_BUFFER: int = (
    300  # seconds, added to per-role timeout for hard safety net
)
NODE_REGEN_MAX: int = 2  # retries for JSON regeneration
WATCHDOG_POLL_INTERVAL: int = 10  # seconds

LONG_TIMEOUT_ROLES: set[str] = {"ARCHITECT", "CODER", "ARCHIVIST_A2"}

# ---------------------------------------------------------------------------
# Execution limits
# ---------------------------------------------------------------------------
RECURSION_LIMIT: int = 50  # enforced by LangGraph runtime
MAX_CORRECTION_ATTEMPTS: int = 4  # correction_attempt > 4 → escalation
MAX_LOOP_ITERATIONS: int = 2  # original + 1 Hard Reset
MAX_PLAN_STEPS: int = 30
MAX_CODE_LOC: int = 5000
MAX_ERROR_LOGS_IN_STATE: int = 20  # older entries rotated to archive
MAX_ARCHIVAL_ATTEMPTS: int = 4
MAX_META_ATTEMPTS: int = 3
MAX_VALIDATION_ATTEMPTS: int = 3
ARCHIVAL_ENABLED: bool = False
MAX_INFRASTRUCTURE_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Telegram Bot Configuration
# ----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")

# Parse allowed users from env var (comma-separated integers)
_allowed_users_str = os.getenv("ALLOWED_TELEGRAM_USERS", "")
ALLOWED_TELEGRAM_USERS: list[int] = []
if _allowed_users_str:
    try:
        ALLOWED_TELEGRAM_USERS = [
            int(u.strip()) for u in _allowed_users_str.split(",") if u.strip()
        ]
    except ValueError:
        pass  # Fail safe to empty list if parsing fails


SANDBOX_CLEANUP_WORKSPACES: bool = True


def _validate():
    """
    Validate critical configuration on import.
    Raises ValueError if configuration is insecure or invalid.
    """
    import re

    # 1. Validate Telegram Token if present
    if TELEGRAM_BOT_TOKEN:
        if not re.match(r"^\d+:[A-Za-z0-9_-]+$", TELEGRAM_BOT_TOKEN):
            raise ValueError(
                f"Invalid TELEGRAM_BOT_TOKEN format: {TELEGRAM_BOT_TOKEN[:5]}..."
            )

    # 2. Validate Allowed Users
    # (Implicitly validated by the parsing logic above which defaults to [] on error)
    pass


_validate()
