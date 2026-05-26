import os
import logging
import datetime
from pathlib import Path
from contextvars import ContextVar

# ANSI Escape Sequences for Terminal Colors
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_GREY = "\033[90m"  # Dark Grey / Bright Black
COLOR_RESET = "\033[0m"

# Thread-safe ContextVars to track active pipeline runs
current_run_id = ContextVar("current_run_id", default="system")
current_project_title = ContextVar("current_project_title", default="")

class RunIdFilter(logging.Filter):
    """
    Injects the active context-bound run_id and project_title into the log record
    so formatters can access them dynamically.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = current_run_id.get()
        record.project_title = current_project_title.get()
        return True

class ColoredConsoleFormatter(logging.Formatter):
    """
    Applies professional ANSI colors to the console output:
    - INFO: Green
    - WARNING: Yellow
    - ERROR / CRITICAL: Red
    - Loggers named 'agents' or starting with 'agents.': Grey
    Also dynamically prefixes logs with the active [run_id] if present.
    """
    def format(self, record: logging.LogRecord) -> str:
        orig_msg = record.msg
        orig_levelname = record.levelname

        # Determine the target color
        color = ""
        if record.name == "agents" or record.name.startswith("agents."):
            color = COLOR_GREY
        elif record.levelno == logging.INFO:
            color = COLOR_GREEN
        elif record.levelno == logging.WARNING:
            color = COLOR_YELLOW
        elif record.levelno >= logging.ERROR:
            color = COLOR_RED

        # Build run ID prefix
        run_id = getattr(record, "run_id", "system")
        run_prefix = f"[{run_id}] " if run_id != "system" else ""

        # Apply color if mapped
        if color:
            record.levelname = f"{color}{record.levelname}{COLOR_RESET}"
            record.msg = f"{color}{run_prefix}{record.msg}{COLOR_RESET}"
        else:
            record.msg = f"{run_prefix}{record.msg}"

        result = super().format(record)

        # Restore original log record attributes
        record.msg = orig_msg
        record.levelname = orig_levelname
        return result

class UncoloredFileFormatter(logging.Formatter):
    """
    Formats log entries for file output without ANSI colors, ensuring
    clear readability and clean text searches in log files.
    """
    def format(self, record: logging.LogRecord) -> str:
        orig_msg = record.msg
        run_id = getattr(record, "run_id", "system")
        run_prefix = f"[{run_id}] " if run_id != "system" else ""
        record.msg = f"{run_prefix}{record.msg}"
        
        result = super().format(record)
        
        record.msg = orig_msg
        return result

def setup_logging():
    """
    Sets up the central logging system:
    - Creates the logs directory.
    - Routes all logs (level >= DEBUG) to logs/app.log.
    - Routes warning & error logs (level >= WARNING) to logs/error.log.
    - Configures console with custom ColoredConsoleFormatter.
    - Attaches a RunIdFilter to all handlers.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Prevent re-adding handlers if logging is already initialized
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.DEBUG)
    
    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredConsoleFormatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(RunIdFilter())
    root_logger.addHandler(console_handler)

    # 2. app.log Handler (All logs)
    app_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    app_handler.setLevel(logging.DEBUG)
    file_formatter = UncoloredFileFormatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
    app_handler.setFormatter(file_formatter)
    app_handler.addFilter(RunIdFilter())
    root_logger.addHandler(app_handler)

    # 3. error.log Handler (Warnings & Errors only)
    error_handler = logging.FileHandler(log_dir / "error.log", encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(file_formatter)
    error_handler.addFilter(RunIdFilter())
    root_logger.addHandler(error_handler)

    # 4. agents.log Handler (Dedicated Agent logger)
    agents_logger = logging.getLogger("agents")
    agents_logger.setLevel(logging.DEBUG)
    
    agents_file_handler = logging.FileHandler(log_dir / "agents.log", encoding="utf-8")
    agents_file_handler.setLevel(logging.DEBUG)
    agents_formatter = UncoloredFileFormatter("%(asctime)s - %(levelname)s - %(message)s")
    agents_file_handler.setFormatter(agents_formatter)
    agents_file_handler.addFilter(RunIdFilter())
    agents_logger.addHandler(agents_file_handler)

def start_new_agent_run(run_id: str, project_title: str):
    """
    Signals the start of a new pipeline execution run:
    - Binds the active run_id and project_title in Thread-safe ContextVars.
    - Appends a prominent separator banner to the logs/agents.log file.
    """
    setup_logging()  # Ensure logging is initialized
    
    current_run_id.set(run_id)
    current_project_title.set(project_title)
    
    agents_logger = logging.getLogger("agents")
    separator = "=" * 80
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = (
        f"\n{separator}\n"
        f"NEW PIPELINE RUN: {run_id} at {timestamp}\n"
        f"Project Title: {project_title}\n"
        f"{separator}\n"
    )
    agents_logger.info(banner)
