"""Terminal logging: see what the agent is doing while the UI runs.

Level via env: LOG_LEVEL=DEBUG shows full tool results; INFO (default) shows short previews.
"""

import logging
import os

FORMAT = "%(asctime)s %(levelname)-5s %(name)-18s | %(message)s"
NOISY = ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "mcp", "urllib3", "langchain_mcp_adapters",
         "asyncio")


def configure_logging() -> None:
    """Configure once (safe to call on every Streamlit rerun)."""
    root = logging.getLogger()
    if getattr(root, "_itsm_configured", False):
        return
    # force=True: importing FastMCP installs its own (rich) root handler; replace it with ours.
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format=FORMAT, datefmt="%H:%M:%S", force=True)
    for name in NOISY:  # keep library chatter out of the demo terminal
        logging.getLogger(name).setLevel(logging.WARNING)
    root._itsm_configured = True  # type: ignore[attr-defined]


def preview(value: object, limit: int = 160) -> str:
    """One-line preview of a tool argument/result (full text at DEBUG level)."""
    text = " ".join(str(value).split())
    if logging.getLogger("itsm").isEnabledFor(logging.DEBUG):
        return text
    return text if len(text) <= limit else text[:limit] + "…"
