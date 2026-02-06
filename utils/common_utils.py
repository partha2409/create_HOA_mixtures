def round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, list):
        return [round_floats(x, ndigits) for x in obj]
    elif isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    else:
        return obj