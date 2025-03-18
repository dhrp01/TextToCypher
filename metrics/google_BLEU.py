import evaluate

def google_BLEU(pred_cypher: str,
               target_cypher: str) -> float:
    """Test semantic similarity of two cyphers using Google-BLEU metric"""
    google_bleu = evaluate.load("google_bleu")
    predictions = [pred_cypher]
    references = [[target_cypher]]
    results = google_bleu.compute(predictions=predictions, references=references)
    return results 