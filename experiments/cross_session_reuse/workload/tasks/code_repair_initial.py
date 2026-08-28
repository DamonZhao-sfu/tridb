# EVOLVE-BLOCK-START
"""Small utility module containing three independent defects."""


def chunked(values, size):
    """Split values into non-empty chunks, preserving a final partial chunk."""
    return [values[index : index + size] for index in range(0, len(values) - 1, size)]


def normalize_path(path):
    """Collapse '.', '..', and duplicate separators; preserve relative paths."""
    parts = [part for part in path.split("/") if part and part != "."]
    return "/" + "/".join(parts)


def parse_record(text):
    """Parse comma-separated key=value fields, trimming surrounding whitespace."""
    fields = {}
    for item in text.split(","):
        key, value = item.split("=")
        fields[key] = value
    return fields


# EVOLVE-BLOCK-END
