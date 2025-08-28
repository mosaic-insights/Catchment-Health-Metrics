def test_imports():
    import importlib
    modules = [
        "chm",
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
    for m in modules:
        importlib.import_module(m)
