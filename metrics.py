import numpy as np


def quadratic_kappa(y, t, eps=1e-15):
    # Assuming y and t are one-hot encoded!
    num_scored_items = y.shape[0]
    num_ratings = y.shape[1]
    ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
                          reps=(1, num_ratings))
    ratings_squared = (ratings_mat - ratings_mat.T) ** 2
    weights = ratings_squared / (float(num_ratings) - 1) ** 2

    # We norm for consistency with other variations.
    y_norm = y / (eps + y.sum(axis=1)[:, None])

    # The histograms of the raters.
    hist_rater_a = y_norm.sum(axis=0)
    hist_rater_b = t.sum(axis=0)

    # The confusion matrix.
    conf_mat = np.dot(y_norm.T, t)

    # The nominator.
    nom = np.sum(weights * conf_mat)
    expected_probs = np.dot(hist_rater_a[:, None],
                            hist_rater_b[None, :])
    # The denominator.
    denom = np.sum(weights * expected_probs / num_scored_items)

    return 1 - nom / denom