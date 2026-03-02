from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.txt"
RESULTS_DIR = BASE_DIR / "results"

TEST_SIZE = 0.2
RANDOM_SEED = 42
LEARNING_RATE = 0.1
MAX_ITERATIONS = 20000
TOLERANCE = 1e-8


def load_dataset(path: Path):
    with path.open("r", encoding="cp1251") as source:
        rows = [line.strip() for line in source if line.strip()]

    header = rows[0].split("\t")
    records = [row.split("\t") for row in rows[1:]]

    data = np.array(records, dtype=np.float64)
    features = data[:, :-1]
    target = data[:, -1].astype(np.int64)
    feature_names = header[:-1]
    target_name = header[-1]
    return features, target, feature_names, target_name


def train_test_split_manual(features, target, test_size=0.2, random_seed=42):
    rng = np.random.default_rng(random_seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)

    test_count = int(round(features.shape[0] * test_size))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        features[train_indices],
        features[test_indices],
        target[train_indices],
        target[test_indices],
    )


def standardize_fit(features):
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds == 0] = 1.0
    return means, stds


def standardize_transform(features, means, stds):
    return (features - means) / stds


def sigmoid(values):
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1, max_iterations=20000, tolerance=1e-8):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None

    def fit(self, features, target):
        sample_count, feature_count = features.shape
        design_matrix = np.hstack((np.ones((sample_count, 1)), features))

        self.weights = np.zeros(feature_count + 1, dtype=np.float64)
        previous_loss = np.inf

        for _ in range(self.max_iterations):
            linear_response = design_matrix @ self.weights
            probabilities = sigmoid(linear_response)

            epsilon = 1e-12
            loss = -np.mean(
                target * np.log(probabilities + epsilon)
                + (1 - target) * np.log(1 - probabilities + epsilon)
            )

            gradient = (design_matrix.T @ (probabilities - target)) / sample_count
            self.weights -= self.learning_rate * gradient

            if abs(previous_loss - loss) < self.tolerance:
                break

            previous_loss = loss

        return self

    def predict_proba(self, features):
        design_matrix = np.hstack((np.ones((features.shape[0], 1)), features))
        return sigmoid(design_matrix @ self.weights)

    def predict(self, features, threshold=0.5):
        probabilities = self.predict_proba(features)
        return (probabilities >= threshold).astype(np.int64)


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def safe_abs_pearson(first, second):
    first_std = np.std(first)
    second_std = np.std(second)
    if first_std == 0 or second_std == 0:
        return 0.0

    correlation = np.corrcoef(first, second)[0, 1]
    if np.isnan(correlation):
        return 0.0

    return float(abs(correlation))


def compute_feature_target_correlations(features, target):
    return np.array(
        [safe_abs_pearson(features[:, index], target) for index in range(features.shape[1])],
        dtype=np.float64,
    )


def compute_feature_feature_correlations(features):
    feature_count = features.shape[1]
    correlations = np.zeros((feature_count, feature_count), dtype=np.float64)

    for row in range(feature_count):
        for column in range(feature_count):
            correlations[row, column] = safe_abs_pearson(
                features[:, row], features[:, column]
            )

    return correlations


def cfs_merit(feature_indices, feature_target_corr, feature_feature_corr):
    subset_size = len(feature_indices)
    mean_feature_class_corr = float(np.mean(feature_target_corr[list(feature_indices)]))

    pair_corr = []
    for first_index, second_index in combinations(feature_indices, 2):
        pair_corr.append(feature_feature_corr[first_index, second_index])

    mean_feature_feature_corr = float(np.mean(pair_corr)) if pair_corr else 0.0
    denominator = np.sqrt(
        subset_size + subset_size * (subset_size - 1) * mean_feature_feature_corr
    )
    if denominator == 0:
        return 0.0

    return (subset_size * mean_feature_class_corr) / denominator


def select_cfs_subset(features, target, selected_size):
    feature_target_corr = compute_feature_target_correlations(features, target)
    feature_feature_corr = compute_feature_feature_correlations(features)

    best_subset = None
    best_merit = -np.inf

    for subset in combinations(range(features.shape[1]), selected_size):
        merit = cfs_merit(subset, feature_target_corr, feature_feature_corr)
        if merit > best_merit:
            best_merit = merit
            best_subset = subset

    return list(best_subset), best_merit, feature_target_corr


def select_naive_subset(feature_target_corr, selected_size):
    ranked = np.argsort(feature_target_corr)[::-1]
    return list(ranked[:selected_size])


def train_and_evaluate(train_x, test_x, train_y, test_y):
    means, stds = standardize_fit(train_x)
    train_scaled = standardize_transform(train_x, means, stds)
    test_scaled = standardize_transform(test_x, means, stds)

    model = LogisticRegressionGD(
        learning_rate=LEARNING_RATE,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
    )
    model.fit(train_scaled, train_y)

    predictions = model.predict(test_scaled)
    accuracy = accuracy_score(test_y, predictions)
    return model, accuracy


def build_correlation_heatmap(features, target, feature_names, target_name, output_path: Path):
    full_matrix = np.column_stack((features, target))
    labels = feature_names + [target_name]
    correlation_matrix = np.corrcoef(full_matrix, rowvar=False)

    plt.figure(figsize=(10, 8))
    image = plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Коэффициент корреляции")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for row in range(correlation_matrix.shape[0]):
        for column in range(correlation_matrix.shape[1]):
            value = correlation_matrix[row, column]
            plt.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.title("Тепловая карта корреляций")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_feature_list(feature_names, feature_indices):
    return ", ".join(feature_names[index] for index in feature_indices)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл данных: {DATA_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features, target, feature_names, target_name = load_dataset(DATA_PATH)
    train_x, test_x, train_y, test_y = train_test_split_manual(
        features,
        target,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
    )

    _, baseline_accuracy = train_and_evaluate(train_x, test_x, train_y, test_y)

    selected_size = features.shape[1] - 2
    cfs_subset, cfs_merit_value, feature_target_corr = select_cfs_subset(
        train_x,
        train_y,
        selected_size=selected_size,
    )
    naive_subset = select_naive_subset(feature_target_corr, selected_size=selected_size)

    _, cfs_accuracy = train_and_evaluate(
        train_x[:, cfs_subset],
        test_x[:, cfs_subset],
        train_y,
        test_y,
    )
    _, naive_accuracy = train_and_evaluate(
        train_x[:, naive_subset],
        test_x[:, naive_subset],
        train_y,
        test_y,
    )

    build_correlation_heatmap(
        features,
        target,
        feature_names,
        target_name,
        RESULTS_DIR / "correlation_heatmap.png",
    )

    metrics_lines = [
        f"Размер выборки: {features.shape[0]} объектов",
        f"Количество признаков: {features.shape[1]}",
        f"Обучающая выборка: {train_x.shape[0]} объектов",
        f"Тестовая выборка: {test_x.shape[0]} объектов",
        "",
        f"Базовая модель (все признаки): {baseline_accuracy:.4f}",
        "",
        "CFS-отбор признаков:",
        f"Выбранные признаки: {format_feature_list(feature_names, cfs_subset)}",
        f"Значение критерия CFS: {cfs_merit_value:.4f}",
        f"Точность: {cfs_accuracy:.4f}",
        "",
        "Наивный отбор признаков:",
        f"Выбранные признаки: {format_feature_list(feature_names, naive_subset)}",
        f"Точность: {naive_accuracy:.4f}",
    ]

    print("\n".join(metrics_lines))
    print(f"\nТепловая карта сохранена в: {RESULTS_DIR / 'correlation_heatmap.png'}")


if __name__ == "__main__":
    main()
