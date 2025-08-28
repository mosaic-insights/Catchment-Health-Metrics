
def app():
    """
    Minimal command-line entry point for Catchment Health Metrics.

    Usage (after installation):
        $ catchment-health-metrics --help
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="catchment-health-metrics",
        description="Run basic CHM checks and print environment info."
    )
    parser.add_argument("--check", action="store_true", help="Run a quick import check for all modules.")
    args = parser.parse_args()

    if args.check:
        modules = [
            "chm.topography",
            "chm.vegetation",
            "chm.connectivity",
            "chm.rusle",
            "chm.hydroclimate_awap",
            "chm.hydroclimate_awral_historical",
            "chm.hydroclimate_awral_projection",
            "chm.landuse",
            "chm.bushfire",
            "chm.roads",
            "chm.monitoring_data",
        ]
        failed = []
        for m in modules:
            try:
                __import__(m)
            except Exception as e:
                failed.append((m, str(e)))
        if failed:
            print("Import check failed for:")
            for m, err in failed:
                print(f" - {m}: {err}")
            sys.exit(1)
        else:
            print("All CHM modules imported successfully.")
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(0)
