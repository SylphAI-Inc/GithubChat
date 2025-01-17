import os
import logging
import inspect
import json
from dotenv import load_dotenv

load_dotenv()

# Global flag for verbose logging
VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "false").lower() == "true"

# Set up root logger
logging.basicConfig(
    level=logging.DEBUG if VERBOSE_LOGS else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)

# Get the root logger
logger = logging.getLogger(__name__)

# Explicitly set the logger's level based on VERBOSE_LOGS
if VERBOSE_LOGS:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Configure adalflow logger
adalflow_logger = logging.getLogger("adalflow")
if VERBOSE_LOGS:
    adalflow_logger.setLevel(logging.DEBUG)
else:
    adalflow_logger.setLevel(logging.WARNING)

# Optional: Add a PrettyLogFormatter for better readability when verbose


class PrettyLogFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
            record.msg = json.dumps(record.msg, indent=4)
        elif isinstance(record.args, dict):
            record.args = json.dumps(record.args, indent=4)
        return super().format(record)


if VERBOSE_LOGS:
    pretty_formatter = PrettyLogFormatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
    )
    for handler in adalflow_logger.handlers:
        handler.setFormatter(pretty_formatter)
    for handler in logger.handlers:
        handler.setFormatter(pretty_formatter)

# Helper functions for "cute" logs


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, type):
        return obj.__name__
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


def get_caller_info():
    """
    Dynamically fetch the caller's module, file, and line number.
    Returns:
        tuple: (caller_module, caller_file, caller_line)
    """
    # Get the caller frame (2 levels up: skip `get_caller_info` and the log function itself)
    frame = inspect.stack()[2]
    module = inspect.getmodule(frame[0])
    file_name = os.path.basename(module.__file__) if module else "Unknown"
    line_number = frame.lineno
    return file_name, line_number

def log_init(component_name: str, config: dict):
    """
    Logs an initialization message with caller information.

    Args:
        component_name (str): The name of the component being initialized.
        config (dict): Configuration for the component.
    """
    file_name, line_number = get_caller_info()
    serializable_config = make_serializable(config)
    pretty_config = json.dumps(serializable_config, indent=4)
    logger.info(f"✨ [{file_name}:{line_number}] {component_name} initialized with configuration:\n{pretty_config}")

def log_info(message: str):
    """
    Logs an informational message with dynamic caller information.
    """
    file_name, line_number = get_caller_info()
    logger.info(f"🔍 [{file_name}:{line_number}] {message}")

def log_success(message: str):
    """
    Logs a success message with dynamic caller information.
    """
    file_name, line_number = get_caller_info()
    logger.info(f"✅ [{file_name}:{line_number}] {message}")

def log_warning(message: str):
    """
    Logs a warning message with dynamic caller information.
    """
    file_name, line_number = get_caller_info()
    logger.warning(f"⚠️ [{file_name}:{line_number}] {message}")

def log_error(message: str):
    """
    Logs an error message with dynamic caller information.
    """
    file_name, line_number = get_caller_info()
    logger.error(f"❌ [{file_name}:{line_number}] {message}")

def log_debug(message: str):
    """
    Logs a debug message with dynamic caller information.
    """
    file_name, line_number = get_caller_info()
    logger.debug(f"🐛 [{file_name}:{line_number}] {message}")

# LoggerUtility remains unchanged


class LoggerUtility:
    """Utility class for advanced logging and directory traversal."""

    @staticmethod
    def traverse_and_log(base_path, ignored_paths, include_extensions, process_file_callback):
        """
        Traverses a directory, logs the directory tree, and optionally processes files.

        Args:
            base_path (str): The root path of the directory to traverse.
            ignored_paths (list): A list of paths to ignore.
            include_extensions (list): A list of file extensions to include.
            process_file_callback (callable): A callback to process each included file.
                                              It should accept arguments (file_path, relative_path, is_code).
        """
        # Use the global logger instead of LoggerUtility.logger
        logger = logging.getLogger(__name__)
        logger.info("Directory tree for %s:", base_path)
        logger.info("project_root/")
        logger.info("|")

        total_files = 0
        for root, dirs, files in os.walk(base_path, topdown=True):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(
                ignored in os.path.join(root, d) for ignored in ignored_paths)]

            # Calculate indentation based on directory depth
            level = root.replace(base_path, '').count(os.sep)
            indent = '|   ' * level
            base_name = os.path.basename(root) or "project_root"

            # Log directory name
            logger.info(f"{indent}|── {base_name}/")
            subindent = '|   ' * (level + 1)

            # Log files in the directory and optionally process them
            for i, name in enumerate(sorted(files)):
                file_path = os.path.join(root, name)
                relative_path = os.path.relpath(file_path, base_path)
                is_last = (i == len(files) - 1)
                prefix = '└── ' if is_last else '├── '
                is_code = os.path.splitext(name)[1] in include_extensions

                if any(ignored in relative_path for ignored in ignored_paths):
                    logger.info(f"{subindent}{prefix}[IGNORED] {name}")
                elif is_code:
                    logger.info(f"{subindent}{prefix}[INCLUDED] {name}")
                    process_file_callback(file_path, relative_path, is_code)
                    total_files += 1
                else:
                    logger.info(f"{subindent}{prefix}[IGNORED] {name}")

        logger.info("Traversal complete. Processed %d files.", total_files)


# Expose the root and adalflow loggers, and LoggerUtility for import
__all__ = [
    "logger",
    "adalflow_logger",
    "LoggerUtility",
    "log_init",
    "log_info",
    "log_success",
    "log_warning",
    "log_error",
    "log_debug",
]



