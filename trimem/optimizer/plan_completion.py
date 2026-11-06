"""One deterministic completion implementation, shared by all proposal sources."""


def complete(config, space):
    return space.canonicalize(config)


def completion_fields(config, space):
    completed = complete(config, space)
    return completed, [
        name
        for name, value in completed.items()
        if value is not None and (name not in config or config[name] is None)
    ]
