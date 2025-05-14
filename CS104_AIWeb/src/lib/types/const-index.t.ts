export const algorithms = [
    { label: "Decision Tree (Custom)", value: "tree_custom" },
    { label: "Decision Tree (sklearn)", value: "tree_sklearn" },
    { label: "Lasso Regression (Custom)", value: "lasso_custom" },
    { label: "Lasso Regression (sklearn)", value: "lasso_sklearn" },
    { label: "ANN (Custom)", value: "ann_custom" },
    { label: "ANN (sklearn)", value: "ann_sklearn" },
] as const;

export const algorithmCodes: Record<string, string> = {
    linear: `def train_and_predict(X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred`,

    tree_custom: `# Custom Decision Tree implementation
class Node:
    def __init__(self, feature_index, threshold, left, right, value):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class Decision_tree:
    def __init__(self, min_samples_split=2, max_depth=2, min_samples_leaf=1, random_state=0):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.rng = np.random.RandomState(random_state)

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _split(self, X, y, feature_index, threshold):
        X_left = X[X[:, feature_index] <= threshold]
        y_left = y[X[:, feature_index] <= threshold]
        X_right = X[X[:, feature_index] > threshold]
        y_right = y[X[:, feature_index] > threshold]
        return X_left, X_right, y_left, y_right

    def _information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return self._mse(parent) - (weight_l  self._mse(l_child) + weight_r  self._mse(r_child))

    def _best_split(self, X, y):
        n, m = X.shape
        if n < self.min_samples_split:
            return None
        list_best_split = []
        max_info = -float("inf")
        for idx in range(m):
            thresholds = np.unique(X[:, idx])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, idx, threshold)
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                info_gain = self._information_gain(y, y_left, y_right)
                if info_gain > max_info:
                    max_info = info_gain
                    current_split = {
                        "feature_index": idx,
                        "threshold": threshold,
                        "X_left": X_left,
                        "X_right": X_right,
                        "y_left": y_left,
                        "y_right": y_right
                    }
                    list_best_split = [current_split]
                elif info_gain == max_info:
                    list_best_split.append(current_split)
        return self.rng.choice(list_best_split) if list_best_split else None

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth:
            return Node(None, None, None, None, np.mean(y))
        best_split = self._best_split(X, y)
        if best_split is None:
            return Node(None, None, None, None, np.mean(y))
        node = Node(best_split["feature_index"], best_split["threshold"], None, None, np.mean(y))
        node.left = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
        node.right = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_one(self, node, x):
        if node.left is None and node.right is None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])
`,

    tree_sklearn: `from sklearn.tree import DecisionTreeRegressor`,

    lasso_custom: `class LassoRegression:
    def __init__(self, lambda_=10, learning_rate=1e-7, n_iterations=1000000, fit_intercept=True):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None
        self.bias = None
        self.fit_intercept = fit_intercept

    def lasso_loss(self, X, y, beta):
        n = len(y)
        residual_sum_of_squares = np.sum((y - (self.bias if self.fit_intercept else 0.0) - X.dot(beta))**2)
        l1_penalty = self.lambda_ * np.sum(np.abs(beta))
        return (1/n) * residual_sum_of_squares + l1_penalty

    def gradient(self, X, y, beta):
        n = len(y)
        grad_rss = -2 * X.T.dot(y - self.bias - X.dot(beta)) / n
        grad_bias = -2 * np.sum(y - self.bias - X.dot(beta)) / n
        grad_l1 = self.lambda_ * np.sign(beta)
        return grad_rss + grad_l1, grad_bias

    def fit(self, X, y):
        n_features = X.shape[1]
        self.beta = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            grad_beta, grad_bias = self.gradient(X, y, self.beta)
            self.beta -= self.learning_rate * grad_beta
            self.bias -= self.learning_rate * grad_bias

    def predict(self, X):
        return X.dot(self.beta) + (self.bias if self.fit_intercept else 0.0)
`,

    lasso_sklearn: `from sklearn.linear_model import Lasso`,
    ann_custom: `
class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=1000, verbose=False):
        """
        layer_sizes: tuple like (n_features, hidden1, ..., hiddenN, n_outputs)
        learning_rate: float, step size for gradient descent
        n_iterations: int, number of training epochs
        verbose: bool, print progress
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose

        # parameters
        self.weights = {}
        self.biases = {}

        # initialize weights and biases (Xavier uniform)
        for l in range(1, len(layer_sizes)):
            in_dim = layer_sizes[l-1]
            out_dim = layer_sizes[l]
            limit = np.sqrt(1. / in_dim)
            self.weights[l] = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
            self.biases[l] = np.zeros(out_dim)

    def _sigmoid(self, x):
        return expit(x)

    def _sigmoid_derivative(self, a):
        return a * (1.0 - a)

    def _feedforward(self, X):
        activations = [X]
        L = len(self.layer_sizes) - 1
        # hidden layers with sigmoid
        for l in range(1, L):
            z = activations[l-1] @ self.weights[l] + self.biases[l]
            a = self._sigmoid(z)
            activations.append(a)
        # output layer: linear activation
        z_out = activations[L-1] @ self.weights[L] + self.biases[L]
        activations.append(z_out)
        return activations

    def _backpropagate(self, X, y):
        activations = self._feedforward(X)
        deltas = {}
        L = len(self.layer_sizes) - 1

        # output delta (linear + MSE)
        error = activations[L] - y
        deltas[L] = error

        # hidden layers
        for l in range(L-1, 0, -1):
            deltas[l] = (deltas[l+1] @ self.weights[l+1].T) * self._sigmoid_derivative(activations[l])

        # update weights & biases (full-batch GD)
        for l in range(1, L+1):
            grad_w = activations[l-1].T @ deltas[l] / X.shape[0]
            grad_b = np.mean(deltas[l], axis=0)
            self.weights[l] -= self.learning_rate * grad_w
            self.biases[l] -= self.learning_rate * grad_b

    def fit(self, X, y):
        # reshape y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for epoch in range(1, self.n_iterations + 1):
            self._backpropagate(X, y)
            if self.verbose and epoch % (self.n_iterations // 10) == 0:
                preds = self.predict(X)
                loss = np.mean((preds - y) ** 2)
                print(f"Epoch {epoch}/{self.n_iterations} - Loss: {loss:.6f}")

    def predict(self, X):
        return self._feedforward(X)[-1]    
`,
    ann_sklearn: `
from sklearn.neural_network import MLPRegressor`,
} as const;

export const dataIntroduction = {
    name: "Crop Yield Prediction",
    description:
        "Crop yield prediction is a crucial application of machine learning in agriculture, helping improve decision-making and manage risks related to food security and climate change. This dataset combines agricultural production with environmental factors from publicly available sources such as the FAO and World Bank.",
    features: [
        {
            name: "Area",
            type: "string",
            description: "Geographic region (e.g., country or farm area)",
        },
        {
            name: "Item",
            type: "string",
            description: "Type of crop (e.g., maize, rice, wheat)",
        },
        { name: "Year", type: "int", description: "Observation year" },
        {
            name: "average_rain_fall_mm_per_year",
            type: "float",
            description: "Average annual rainfall in millimeters",
        },
        {
            name: "pesticides_tonnes",
            type: "float",
            description: "Amount of pesticide applied (in metric tonnes)",
        },
        {
            name: "avg_temp",
            type: "float",
            description: "Mean annual temperature (in °C)",
        },
    ],
    target: {
        name: "hg/ha_yield",
        type: "float",
        description: "Crop yield measured in hectograms per hectare",
    },
    metrics: [
        { name: "MAE", description: "Mean Absolute Error" },
        { name: "MSE", description: "Mean Squared Error" },
        { name: "R² Score", description: "Coefficient of Determination" },
    ],
    statistics: [
        { name: "Number of samples", value: 28242 },
        { name: "Number of features", value: 7 },
    ],
    sample: [
        {
            Area: "Vietnam",
            Item: "Rice",
            Year: 2010,
            average_rain_fall_mm_per_year: 1800,
            pesticides_tonnes: 200,
            avg_temp: 27.5,
            hg_ha_yield: 55000,
        },
        {
            Area: "India",
            Item: "Wheat",
            Year: 2012,
            average_rain_fall_mm_per_year: 1000,
            pesticides_tonnes: 150,
            avg_temp: 25.0,
            hg_ha_yield: 40000,
        },
    ],
} as const;

export const countryList = [
    "Albania",
    "Algeria",
    "Angola",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Belarus",
    "Belgium",
    "Botswana",
    "Brazil",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cameroon",
    "Canada",
    "Central African Republic",
    "Chile",
    "Colombia",
    "Croatia",
    "Denmark",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Eritrea",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Ghana",
    "Greece",
    "Guatemala",
    "Guinea",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hungary",
    "India",
    "Indonesia",
    "Iraq",
    "Ireland",
    "Italy",
    "Jamaica",
    "Japan",
    "Kazakhstan",
    "Kenya",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Libya",
    "Lithuania",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Mali",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Namibia",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Norway",
    "Pakistan",
    "Papua New Guinea",
    "Peru",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Rwanda",
    "Saudi Arabia",
    "Senegal",
    "Slovenia",
    "South Africa",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Tajikistan",
    "Thailand",
    "Tunisia",
    "Turkey",
    "Uganda",
    "Ukraine",
    "United Kingdom",
    "Uruguay",
    "Zambia",
    "Zimbabwe",
] as const;

export const itemList = [
    "Maize",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Wheat",
    "Cassava",
    "Sweet potatoes",
    "Plantains and others",
    "Yams",
] as const;

export const HomePageContent = {
    title: "Crop Yield Prediction Using Machine Learning",
    description:
        "A full ML solution that predicts crop yields based on environmental and agricultural data. Includes modeling, evaluation, and result visualization.",
    aboutProject: `This project aims to design, implement, and evaluate a complete machine learning solution using a real-world dataset in one of four key domains—Education, Agriculture, Healthcare, or Environmental Sustainability. We will develop an end-to-end pipeline that includes data preprocessing, feature engineering, model selection and tuning, and rigorous performance evaluation. To showcase our work, we will build a dedicated website, prepare a slide deck for an in-class presentation, and deliver a 3–4-page written report covering the introduction, problem setup, methodology, experiments, conclusions, and individual contributions. Through this holistic approach, we seek not only to demonstrate technical proficiency with modern ML tools but also to communicate our findings clearly to both technical and non-technical audiences.`,

    teamMembers: [
        { name: "Pham Huynh Long Vu", id: "23521813" },
        { name: "Nguyen Minh Huy", id: "23520634" },
        { name: "Dang Quoc Cuong", id: "23520192" },
    ],

    coreFeature: [
        {
            title: "Dataset",
            items: [
                {
                    name: "Dataset name",
                    desc: "Crop Yield Prediction (source: FAO & World Bank)",
                },
                { name: "Samples", desc: "56,717 samples" },
                {
                    name: "Features",
                    desc: "Area, Item, Year, Rainfall, Temperature, Pesticide",
                },
                {
                    name: "Target",
                    desc: "hg/ha_yield (crop yield per hectare)",
                },
                {
                    name: "EDA",
                    desc: "Histograms, time-based trends, correlation heatmap",
                },
                {
                    name: "Data processing",
                    desc: "Filter countries with < 100 samples, Label & One-Hot Encoding",
                },
            ],
        },
        {
            title: "Algorithm",
            items: [
                {
                    name: "Goal",
                    desc: "Predict crop yield using environmental and agricultural data",
                },
                {
                    name: "Algorithms used",
                    desc: "Decision Tree, Lasso Regression, Linear Regression",
                },
                { name: "Evaluation metrics", desc: "MSE, MAE, R²-score" },
                {
                    name: "Output",
                    desc: "Performance comparison tables by model",
                },
            ],
        },
        {
            title: "Test",
            items: [
                {
                    name: "Prediction",
                    desc: "Load trained models to predict yield on new input",
                },
                {
                    name: "Input methods",
                    desc: "Manual data entry or file upload",
                },
                {
                    name: "Comparison",
                    desc: "Evaluate algorithm performance on the same test samples",
                },
            ],
        },
    ],
};
