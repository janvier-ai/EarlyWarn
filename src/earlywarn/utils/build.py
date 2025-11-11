import importlib

def build_from_path(path: str, **kwargs):
    mod_path, _, attr = path.rpartition(".")
    if not mod_path:
        raise ValueError(f"Expected 'module.submodule.Class', got '{path}'")
    mod = importlib.import_module(mod_path)
    try:
        ctor = getattr(mod, attr)
    except AttributeError:
        raise ValueError(f"'{attr}' not found in module '{mod_path}'")
    return ctor(**kwargs)

def build_from_cfg(cfg):
    return build_from_path(cfg["path"], **cfg.get("params", {}))
