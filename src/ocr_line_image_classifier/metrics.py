import pandas as pd


def calculate_metrics(transcript_df, similarity_threshold):
    """Calculate the true positives, false positives, true negatives, and false negatives."""
    tp = fp = tn = fn = 0

    # Ensure all ocr_text entries are strings and handle missing values
    transcript_df["ocr_text"] = transcript_df["ocr_text"].fillna("").astype(str)

    for _, row in transcript_df.iterrows():
        similarity_score = row["similarity_score"]
        ocr_text = row["ocr_text"].strip()

        has_newline_in_middle = "\n" in ocr_text

        if similarity_score >= similarity_threshold:
            if not has_newline_in_middle:
                tp += 1
            else:
                fp += 1
        else:
            if has_newline_in_middle:
                tn += 1
            else:
                fn += 1

    return tp, fp, tn, fn


def calculate_precision(tp, fp):
    """Calculate the precision score."""
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    """Calculate the recall score."""
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def calculate_f1_score(precision, recall):
    """Calculate the F1 score."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_transcripts(transcript_df, similarity_threshold):
    """Evaluate the transcript data using the similarity threshold."""
    tp, fp, tn, fn = calculate_metrics(transcript_df, similarity_threshold)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1_score = calculate_f1_score(precision, recall)
    stats = [tp, fp, tn, fn, precision, recall, f1_score]
    return stats


def save_evaluation_metrics(stats, output_file):
    """Save the evaluation metrics to a CSV file."""
    tp, fp, tn, fn, precision, recall, f1_score = stats
    metrics_df = pd.DataFrame(
        {
            "true_positives": [tp],
            "false_positives": [fp],
            "true_negatives": [tn],
            "false_negatives": [fn],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1_score],
        }
    )
    metrics_df.to_csv(output_file, index=False)


def classify_transcripts(transcript_df, similarity_threshold):
    """Classify transcript rows based on similarity score and move rows with newlines in OCR text to low similarity."""
    # Ensure all ocr_text entries are strings and handle missing values
    transcript_df["ocr_text"] = (
        transcript_df["ocr_text"].fillna("").astype(str).str.strip()
    )
    transcript_df["pre_processsed_ocr_text"] = (
        transcript_df["pre_processsed_ocr_text"].fillna("").astype(str).str.strip()
    )

    high_similarity_rows = transcript_df[
        (transcript_df["similarity_score"] >= similarity_threshold)
        & (~transcript_df["pre_processsed_ocr_text"].str.contains("\n"))
    ]
    low_similarity_rows = transcript_df[
        (transcript_df["similarity_score"] < similarity_threshold)
        | (transcript_df["pre_processsed_ocr_text"].str.contains("\n"))
    ]
    return high_similarity_rows, low_similarity_rows
