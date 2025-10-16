import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Union
from urllib.parse import unquote

import cv2
import numpy as np
import torch
import tqdm

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
LOGGING_NAME = "vidnn"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = torch.__version__
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # faster than importing torchvision
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"

# Settings and Environment Variables
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter=dict(float_kind="{:11.5g}".format))  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs


class DataExportMixin:
    """
    Mixin class for exporting validation metrics or prediction results in various formats.

    This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results
    from classification, object detection, segmentation, or pose estimation tasks into various formats: Pandas
    DataFrame, CSV, XML, HTML, JSON and SQLite (SQL).

    Methods:
        to_df: Convert summary to a Pandas DataFrame.
        to_csv: Export results as a CSV string.
        to_xml: Export results as an XML string (requires `lxml`).
        to_html: Export results as an HTML table.
        to_json: Export results as a JSON string.
        tojson: Deprecated alias for `to_json()`.
        to_sql: Export results to an SQLite database.

    Examples:
        >>> model = YOLO("yolo11n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
        >>> results.to_sql(table_name="yolo_results")
    """

    def to_df(self, normalize=False, decimals=5):
        """
        Create a pandas DataFrame from the prediction results summary or validation metrics.

        Args:
            normalize (bool, optional): Normalize numerical values for easier comparison.
            decimals (int, optional): Decimal places to round floats.

        Returns:
            (DataFrame): DataFrame containing the summary data.
        """
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """
        Export results to CSV string format.

        Args:
           normalize (bool, optional): Normalize numeric values.
           decimals (int, optional): Decimal precision.

        Returns:
           (str): CSV content as string.
        """
        return self.to_df(normalize=normalize, decimals=decimals).to_csv()

    def to_xml(self, normalize=False, decimals=5):
        """
        Export results to XML format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): XML string.

        Notes:
            Requires `lxml` package to be installed.
        """
        df = self.to_df(normalize=normalize, decimals=decimals)
        return '<?xml version="1.0" encoding="utf-8"?>\n<root></root>' if df.empty else df.to_xml(parser="etree")

    def to_html(self, normalize=False, decimals=5, index=False):
        """
        Export results to HTML table format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.
            index (bool, optional): Whether to include index column in the HTML table.

        Returns:
            (str): HTML representation of the results.
        """
        df = self.to_df(normalize=normalize, decimals=decimals)
        return "<table></table>" if df.empty else df.to_html(index=index)

    def tojson(self, normalize=False, decimals=5):
        """Deprecated version of to_json()."""
        LOGGER.warning("'result.tojson()' is deprecated, replace with 'result.to_json()'.")
        return self.to_json(normalize, decimals)

    def to_json(self, normalize=False, decimals=5):
        """
        Export results to JSON format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): JSON-formatted string of the results.
        """
        return self.to_df(normalize=normalize, decimals=decimals).to_json(orient="records", indent=2)

    def to_sql(self, normalize=False, decimals=5, table_name="results", db_path="results.db"):
        """
        Save results to an SQLite database.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.
            table_name (str, optional): Name of the SQL table.
            db_path (str, optional): SQLite database file path.
        """
        df = self.to_df(normalize, decimals)
        if df.empty or df.columns.empty:  # Exit if df is None or has no columns (i.e., no schema)
            return

        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Dynamically create table schema based on summary to support prediction and validation results export
        columns = []
        for col in df.columns:
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample_val, dict):
                col_type = "TEXT"
            elif isinstance(sample_val, (float, int)):
                col_type = "REAL"
            else:
                col_type = "TEXT"
            columns.append(f'"{col}" {col_type}')  # Quote column names to handle special characters like hyphens

        # Create table (Drop table from db if it's already exist)
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cursor.execute(f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {", ".join(columns)})')

        for _, row in df.iterrows():
            values = [json.dumps(v) if isinstance(v, dict) else v for v in row]
            column_names = ", ".join(f'"{col}"' for col in df.columns)
            placeholders = ", ".join("?" for _ in df.columns)
            cursor.execute(f'INSERT INTO "{table_name}" ({column_names}) VALUES ({placeholders})', values)

        conn.commit()
        conn.close()
        LOGGER.info(f"Results saved to SQL table '{table_name}' in '{db_path}'.")


class SimpleClass:
    """
    A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings,
    showing all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Return a human-readable string representation of the object.
        __repr__: Return a machine-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


def set_logging(name="LOGGING_NAME", verbose=True):
    """
    Set up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and
    formatter based on the verbosity flag and the current process rank. It handles special cases for Windows
    environments where UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger.
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise.

    Returns:
        (logging.Logger): Configured logger object.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    class PrefixFormatter(logging.Formatter):
        def format(self, record):
            """Format log records with prefixes based on level."""
            # Apply prefixes based on log level
            if record.levelno == logging.WARNING:
                prefix = "WARNING ⚠️" if not WINDOWS else "WARNING"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR ❌" if not WINDOWS else "ERROR"
                record.msg = f"{prefix} {record.msg}"

            # Handle emojis in message based on platform
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")

    # Handle Windows UTF-8 encoding issues
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class for handling exceptions gracefully.

    This class can be used as a decorator or context manager to catch exceptions and optionally print warning messages.
    It allows code to continue execution even when exceptions occur, which is useful for non-critical operations.

    Attributes:
        msg (str): Optional message to display when an exception occurs.
        verbose (bool): Whether to print the exception message.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>> # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Execute when entering TryExcept context, initialize instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Define behavior when exiting a 'with' block, print error message if necessary."""
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True


def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Args:
        rcparams (dict, optional): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend.

    Examples:
        >>> @plt_settings({"font.size": 12})
        >>> def plot_function():
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()

        >>> with plt_settings({"font.size": 12}):
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Set rc parameters and backend, call the original function, and restore the settings."""
            import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            # Plot with backend and always revert to original backend
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def check_configs(cfg):
    if cfg["single_cls"]:
        cfg["names"] = {0: "item"}

    return cfg
