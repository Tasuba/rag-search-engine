def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"vectors must be the same length (got {len(vec1)} and {len(vec2)})"
        )
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] + vec2[i])
    return result


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"vectors must be the same length (got {len(vec1)} and {len(vec2)})"
        )
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] - vec2[i])
    return result
