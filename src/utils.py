"""
Stock Analyzer - Utilities
Helper functions for formatting, file I/O, and watchlist management.
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ─── Logging setup ───────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    """Configure application-wide logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


# ─── Watchlist Management ───────────────────────────────────────────────────

def load_watchlist(name: str = "default") -> list[str]:
    """Load a watchlist from JSON file."""
    os.makedirs(config.WATCHLIST_DIR, exist_ok=True)
    path = os.path.join(config.WATCHLIST_DIR, f"{name}.json")

    if not os.path.exists(path):
        if name == "default":
            # Create default watchlist
            save_watchlist(config.DEFAULT_UNIVERSE[:20], "default")
            return config.DEFAULT_UNIVERSE[:20]
        logger.warning("Watchlist '%s' not found", name)
        return []

    with open(path, "r") as f:
        data = json.load(f)
    return data.get("symbols", [])


def save_watchlist(symbols: list[str], name: str = "default"):
    """Save a watchlist to JSON file."""
    os.makedirs(config.WATCHLIST_DIR, exist_ok=True)
    path = os.path.join(config.WATCHLIST_DIR, f"{name}.json")

    data = {
        "name": name,
        "updated": datetime.now().isoformat(),
        "symbols": [s.upper().strip() for s in symbols],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Watchlist '%s' saved with %d symbols", name, len(symbols))


def list_watchlists() -> list[str]:
    """List available watchlist names."""
    os.makedirs(config.WATCHLIST_DIR, exist_ok=True)
    return [
        f.replace(".json", "")
        for f in os.listdir(config.WATCHLIST_DIR)
        if f.endswith(".json")
    ]


def add_to_watchlist(symbols: list[str], name: str = "default"):
    """Add symbols to an existing watchlist."""
    current = load_watchlist(name)
    new_syms = [s.upper().strip() for s in symbols if s.upper().strip() not in current]
    if new_syms:
        current.extend(new_syms)
        save_watchlist(current, name)
        logger.info("Added %s to watchlist '%s'", new_syms, name)


def remove_from_watchlist(symbols: list[str], name: str = "default"):
    """Remove symbols from a watchlist."""
    current = load_watchlist(name)
    to_remove = {s.upper().strip() for s in symbols}
    updated = [s for s in current if s not in to_remove]
    save_watchlist(updated, name)
    logger.info("Removed %s from watchlist '%s'", to_remove, name)


# ─── Display Helpers ─────────────────────────────────────────────────────────

def print_header(text: str):
    """Print a styled header."""
    width = max(60, len(text) + 4)
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


def print_divider():
    print(f"{'─' * 60}")


def progress_bar(current: int, total: int, symbol: str = "", width: int = 30):
    """Print a progress bar (updates in-place)."""
    pct = current / total
    filled = int(width * pct)
    bar = f"[{'█' * filled}{'░' * (width - filled)}]"
    print(f"\r  {bar} {current}/{total} {symbol:8s}", end="", flush=True)
    if current == total:
        print()  # newline at end


def format_number(val, prefix: str = "", suffix: str = "") -> str:
    """Format large numbers with K/M/B suffixes."""
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"{prefix}{val / 1e12:.2f}T{suffix}"
    if abs(val) >= 1e9:
        return f"{prefix}{val / 1e9:.2f}B{suffix}"
    if abs(val) >= 1e6:
        return f"{prefix}{val / 1e6:.2f}M{suffix}"
    if abs(val) >= 1e3:
        return f"{prefix}{val / 1e3:.1f}K{suffix}"
    return f"{prefix}{val:.2f}{suffix}"


# ─── Report Export ───────────────────────────────────────────────────────────

def save_report(content: str, filename: str):
    """Save a text report to the output directory."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📄 Report saved: {path}")
    return path
