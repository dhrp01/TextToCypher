def exact_match(pred_cypher: str, target_cypher: str) -> float:
    """Test semantic similarity of two cyphers using exact match metric.
    Returns:
        1.0 if the predicted cipher exactly matches the target cipher, otherwise 0.0.
    """
    return 1.0 if pred_cypher == target_cypher else 0.0
