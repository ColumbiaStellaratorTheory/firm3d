import builtins
import logging
import math
import os
import sys

from .mpi import verbose

in_github_actions = os.getenv("GITHUB_ACTIONS") == "true"


def print(*args, **kwargs):
    r"""
    Overloaded print function to force flushing of stdout.
    All procs prints to stdout.
    """
    builtins.print(*args, **kwargs, flush=True, file=sys.stdout)


def proc0_print(*args, **kwargs):
    r"""
    Overloaded print function to force flushing of stdout.
    Only proc0 prints to stdout.
    """
    if verbose:
        builtins.print(*args, **kwargs, flush=True, file=sys.stdout)


def setup_logging(filename):
    r"""
    Setup logging to redirect output to a file while maintaining console output.

    This function configures Python's logging system to write all output to both
    the specified file and the console. It also redirects the built-in print()
    function to use logging, ensuring that all print statements are captured.

    Args:
        filename (str): The path to the log file where output will be written.
                       The file will be opened in append mode.

    Example:
        >>> from firm3d.util.functions import setup_logging
        >>> setup_logging("my_output.log")
        >>> print("This will go to both console and file")
        >>> proc0_print("This will also be captured")
    """
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # Override any existing logging configuration
    )

    # Redirect the built-in print function to use logging
    original_print = builtins.print

    def logged_print(*args, **kwargs):
        # Convert all arguments to strings and join them
        message = " ".join(map(str, args))
        logging.info(message)
        # Also call the original print to maintain console output
        # Handle potential conflict with flush parameter
        if "flush" in kwargs:
            # If flush is already specified, don't add it again
            original_print(*args, **kwargs)
        else:
            # Add flush=True if not already specified
            original_print(*args, **kwargs, flush=True)

    builtins.print = logged_print


def sigmav(T):
    r"""
    Approximate the D-T fusion reactivity <sigma v> as a function of ion
    temperature, using the fitting formula from Bosch & Hale (1992), reduced
    to its leading-order (Gamow-peak) behavior.

    Args:
        T : Ion temperature in keV.

    Returns:
        sigmav : Approximate fusion reactivity (up to an overall constant),
                 or 0 if T <= 0.
    """
    if T > 0:
        return T ** (-2 / 3) * math.exp(-19.94 * T ** (-1 / 3))
    return 0
