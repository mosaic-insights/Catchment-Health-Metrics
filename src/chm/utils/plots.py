# src/chm/utils/plots.py
from pathlib import Path
from IPython.display import Image, display

def show_site_plot(workspace: str | Path, catchment_name: str, site_id: int, layer: str = "DEM"):
    """
    Display a saved site plot (PNG) inside a Jupyter notebook.

    Parameters
    ----------
    workspace : str | Path
        Root CHM workspace directory
    catchment_name : str
        Catchment folder name, e.g. "Example catchment"
    site_id : int
        Site identifier, e.g. 2
    layer : str
        Which layer's plot to show, e.g. "DEM", "RUSLE", "NDVI"
    """
    plots_dir = Path(workspace) / catchment_name / "Sites Plots and Maps" / f"Site_{site_id}"
    plot_file = plots_dir / f"{layer}.png"

    if plot_file.exists():
        display(Image(filename=plot_file, width=600))
    else:
        raise FileNotFoundError(f"Plot not found: {plot_file}")
