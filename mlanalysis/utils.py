
def transpose_nested_dict(nested_dict):
    """
    Transpose a nested dictionary.

    Args:
        nested_dict (dict): A nested dictionary of the form {key1: {key2: value}}.

    Returns:
        dict: A transposed nested dictionary of the form {key2: {key1: value}}.
    """

    nested_dict_T = {
        key2: {key1: nested_dict[key1][key2] for key1 in nested_dict}
        for key2 in next(iter(nested_dict.values()))
    }
    return nested_dict_T
