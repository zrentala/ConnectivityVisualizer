def update_attributes(obj, **kwargs) -> None:
    """
    Update object attributes using keyword arguments.
    If the object defines a method named `update_<key>`, that method is called.
    Otherwise, setattr(obj, key, value) is used.

    Example:
        update_attributes(
            visualizer,
            viz_type="3d",
            conn_min=0.2,
            threshold=0.5,
        )
    """

    for key, value in kwargs.items():
        if value is None:
            continue

        update_method_name = f"update_{key}"

        # Case 1: class has a custom update_<key>() method
        if hasattr(obj, update_method_name):
            method = getattr(obj, update_method_name)
            if callable(method):
                method(value)
                continue

        # Case 2: fall back to plain attribute update
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise AttributeError(
                f"{obj.__class__.__name__} has no attribute or updater for '{key}'"
            )
