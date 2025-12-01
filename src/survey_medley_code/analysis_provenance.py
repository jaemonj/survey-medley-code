import inspect
import json
import os
import socket
import subprocess
import sys
from datetime import datetime

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def run_cmd(cmd, cwd=None):
    try:
        return (
            subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
            .decode()
            .strip()
        )
    except Exception:
        return None


def find_git_repo(path):
    """Walk upward from 'path' to find a git repo root."""
    path = os.path.abspath(path)
    prev = None

    while path != prev:
        if os.path.isdir(os.path.join(path, '.git')):
            return path
        prev = path
        path = os.path.dirname(path)

    return None


def get_git_info(start_path):
    """Return git commit, dirty state, branch."""
    repo = find_git_repo(start_path)
    if not repo:
        return None

    commit = run_cmd(['git', 'rev-parse', 'HEAD'], cwd=repo)
    branch = run_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo)

    try:
        subprocess.check_call(
            ['git', 'diff', '--quiet'],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = False
    except subprocess.CalledProcessError:
        dirty = True

    return {
        'repo_root': repo,
        'commit': commit,
        'branch': branch,
        'dirty': dirty,
    }


def in_jupyter_notebook():
    """Detect if code runs inside a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        if 'IPython' in str(type(shell)):
            if shell.__class__.__name__ == 'ZMQInteractiveShell':
                return True
        return False
    except Exception:
        return False


def get_call_origin():
    """
    Identify where log_provenance() was called from:
    - If notebook → return notebook type and cell info if possible
    - If script → return path to script
    - If interactive → mark it
    """

    if in_jupyter_notebook():
        return {'environment': 'notebook'}

    # Inspect call stack
    stack = inspect.stack()

    # Level 0 is here, level 1 is log_provenance(), level 2 is the caller
    if len(stack) >= 3:
        caller = stack[2]
        return {
            'environment': 'python-script',
            'file': os.path.abspath(caller.filename),
            'function': caller.function,
            'lineno': caller.lineno,
        }

    # Fallback
    return {'environment': 'interactive'}


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------


def log_provenance(output_dir, settings=None):
    """
    Capture:
    - call path (script or notebook)
    - function
    - git repo status
    - environment metadata
    - any user settings
    """
    os.makedirs(output_dir, exist_ok=True)

    provenance = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'hostname': socket.gethostname(),
        'python': sys.version,
        'working_directory': os.getcwd(),
        'call_origin': get_call_origin(),
        'settings': settings or {},
    }

    git_info = get_git_info(os.getcwd())
    provenance['git'] = git_info if git_info else 'not a git repository'

    # Write output
    out_path = os.path.join(output_dir, 'provenance.json')
    with open(out_path, 'w') as f:
        json.dump(provenance, f, indent=2)

    return out_path
