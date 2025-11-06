# src/chm/utils/paths.py
from pathlib import Path
import os

def get_project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up from `start` until
    we find a marker (pyproject.toml or .git). Falls back to start/cwd.
    Env override: CHM_PROJECT_ROOT.
    """
    env_root = os.getenv("CHM_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    cur = (start or Path.cwd()).resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return (start or Path.cwd()).resolve()

def default_paths(
    catchment_name: str = "Example_catchment.gpkg",
    sites_name: str = "Example_sites.gpkg",
    workspace_subdir: str = "tests/Output",
    input_subdir: str = "tests/Input data",
) -> tuple[str, str, str]:
    """Build default paths relative to project root, with env-var overrides:
       - CHM_WORKSPACE: absolute path to workspace directory
       - CHM_INPUT_DIR: absolute path to input data directory
       Returns (workspace, catchment_path, sites_path).
    """
    root = get_project_root()

    input_dir = Path(os.getenv("CHM_INPUT_DIR", str(root / input_subdir))).resolve()
    workspace = Path(os.getenv("CHM_WORKSPACE", str(root / workspace_subdir))).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    catchment = input_dir / catchment_name
    sites = input_dir / sites_name
    return str(workspace), str(catchment), str(sites)
