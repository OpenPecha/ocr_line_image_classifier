import pandas as pd

from ocr_line_image_classifier.metrics import calculate_metrics


def test_calculate_metrics():

    # Test case data
    data = {
        "similarity_score": [0.95, 0.85, 0.92, 0.75, 0.98],
        "ocr_text": [
            "\nThis is a test\n\f",
            "Another\ntest",
            "More \n tests",
            "Final test\n\f",
            "This is a big test",
        ],
        "pre_processed_ocr_text": [
            "This is a test",
            "Another \n test",
            "More tests",
            "Final test",
            "This is \n a big test",
        ],
    }
    transcript_df = pd.DataFrame(data)
    similarity_threshold = 0.9
    tp, fp, tn, fn = calculate_metrics(transcript_df, similarity_threshold)
    assert tp == 2
    assert fp == 1
    assert tn == 1
    assert fn == 1
