"""
Interactive CLI for the AppleSupport AI Agent.

Usage:
    # Interactive mode (REPL)
    python cli.py

    # Single query mode
    python cli.py --query "My iPhone battery drains too fast after the update"

    # Use a custom knowledge base path
    python cli.py --kb data/knowledge_base.json
"""

import argparse
import sys
import os
import textwrap

from agent import SupportAgent, get_best_provider


# ── ANSI colors ──────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"


def _enable_ansi_windows():
    """Enable ANSI / VT100 escape codes on Windows 10+."""
    if os.name != 'nt':
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(-11)
        # Get current mode
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        # If it fails, disable colors entirely so output stays readable
        global BOLD, DIM, RESET, CYAN, GREEN, YELLOW, RED, MAGENTA, WHITE
        BOLD = DIM = RESET = CYAN = GREEN = YELLOW = RED = MAGENTA = WHITE = ""


_enable_ansi_windows()


BANNER = f"""{CYAN}{BOLD}
+------------------------------------------------------------+
|             AppleSupport AI Agent  CLI                      |
|                                                            |
|  Type a customer query and get:                            |
|    - Intent classification                                 |
|    - A drafted reply grounded in historical support data    |
|    - An escalation recommendation                          |
|                                                            |
|  Commands:  /help  /quit  /clear                           |
+------------------------------------------------------------+
{RESET}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    """Word-wrap text with a hanging indent."""
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def print_result(result: dict) -> None:
    """Pretty-print a single agent response."""
    intent = result["intent"]
    reply = result["drafted_reply"] or "(no reply generated)"
    escalate = result["escalate"]
    reason = result["escalation_reason"]

    print()
    print(f"  {BOLD}{CYAN}Intent:{RESET}  {WHITE}{intent}{RESET}")
    print()
    print(f"  {BOLD}{GREEN}Drafted Reply:{RESET}")
    # Print each paragraph separately so multi-line replies render correctly
    for paragraph in reply.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            print(wrap(paragraph))
    print()

    if escalate:
        tag = f"{RED}{BOLD}[!] ESCALATE{RESET}"
    else:
        tag = f"{GREEN}{BOLD}[OK] Auto-handle{RESET}"

    print(f"  {BOLD}{YELLOW}Escalation:{RESET}  {tag}")
    print(f"  {DIM}Reason: {reason}{RESET}")
    print(f"  {DIM}{'-' * 60}{RESET}")


def print_help() -> None:
    print(f"""
  {BOLD}Available commands:{RESET}
    {CYAN}/help{RESET}   - Show this help message
    {CYAN}/quit{RESET}   - Exit the CLI  (also: /exit, Ctrl+C)
    {CYAN}/clear{RESET}  - Clear the terminal screen

  Just type any customer support query at the prompt and press Enter.
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def build_agent(kb_path: str) -> SupportAgent:
    """Initialize the LLM provider and SupportAgent."""
    print(f"{DIM}  Selecting LLM provider...{RESET}", end=" ", flush=True)
    provider = get_best_provider()
    provider_name = type(provider).__name__
    print(f"{GREEN}{provider_name}{RESET}")

    print(f"{DIM}  Loading knowledge base from {kb_path}...{RESET}", end=" ", flush=True)
    agent = SupportAgent(provider, kb_path)
    kb_size = len(agent.knowledge_base)
    idx_size = len(agent.kb_documents) if agent.kb_documents else 0
    print(f"{GREEN}{kb_size} threads, {idx_size} indexed pairs{RESET}")
    print()
    return agent


def run_interactive(agent: SupportAgent) -> None:
    """Run the interactive REPL loop."""
    print(BANNER)
    print_help()

    while True:
        try:
            query = input(f"{BOLD}{MAGENTA}  >> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}  Goodbye!{RESET}")
            break

        if not query:
            continue

        # Slash commands
        lower = query.lower()
        if lower in ("/quit", "/exit"):
            print(f"{DIM}  Goodbye!{RESET}")
            break
        if lower == "/help":
            print_help()
            continue
        if lower == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            continue

        # Process the query
        print(f"\n{DIM}  Processing...{RESET}")
        try:
            result = agent.handle_tweet(query)
            print_result(result)
        except Exception as e:
            print(f"\n  {RED}Error: {e}{RESET}\n")


def run_single(agent: SupportAgent, query: str) -> None:
    """Handle a single query and exit."""
    result = agent.handle_tweet(query)
    print_result(result)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the AppleSupport AI Agent."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single query instead of interactive mode.",
    )
    parser.add_argument(
        "--kb",
        type=str,
        default="data/knowledge_base.json",
        help="Path to the knowledge base JSON file (default: data/knowledge_base.json).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.kb):
        print(f"{RED}Knowledge base not found at '{args.kb}'.{RESET}")
        print(f"Run {CYAN}python data_pipeline.py{RESET} first to generate it.")
        sys.exit(1)

    agent = build_agent(args.kb)

    if args.query:
        run_single(agent, args.query)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
