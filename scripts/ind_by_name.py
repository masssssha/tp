def get_ind(data: list, name: str) -> int:
    """Return the column index by the header from the first row"""
    return (data[0]).index(name)
