![cPMML](./docsrc/img/logo.png)

![GitHub](https://img.shields.io/github/license/AmadeusITGroup/cPMML?style=flat-square)
![](https://img.shields.io/badge/STD-C%2B%2B20-blue?style=flat-square)

# High-Performance PMML Scoring

*cPMML* is a C++ library for scoring and forecasting machine learning models serialized with the Predictive Model Markup Language ([PMML 4.4](http://dmg.org/pmml/v4-4/GeneralStructure.html)).
It exposes a minimalist and user-friendly API targeting high performance, with a predictable and minimal memory footprint.

## Supported Models

### Scoring models (`model.predict` / `model.score`)

| PMML Element | Description |
|---|---|
|`AssociationModel`| Supports all three rule-selection algorithms (recommendation, exclusiveRecommendation, ruleAssociation)|
| `TreeModel` | Decision trees |
| `MiningModel` | Ensembles (random forests, gradient boosting, stacking) |
| `RegressionModel` | Linear and logistic regression |
| `GeneralRegressionModel` | GLMs (linear, multinomial logistic, ordinal multinomial, Poisson, gamma, tweedie) |
| `NeuralNetwork` | Feed-forward neural networks |
| `SupportVectorMachineModel` | SVM (classification and regression) |
| `NearestNeighborModel` | k-NN |
| `NaiveBayesModel` | Naive Bayes |
| `ClusteringModel` | k-means clustering |
| `RuleSetModel` | Rule sets |
| `Scorecard` | Scorecards |
| `BaselineModel` | Baseline / majority class |
| `AnomalyDetectionModel` | Isolation forest anomaly detection |
| `GaussianProcessModel` | Gaussian process regression |
| `TextModel` | Text classification (TF-IDF) |

### Time series models (`model.forecast`)

| `bestFit` value | Algorithm | Notes |
|---|---|---|
| `ExponentialSmoothing` | ETS | All 15 trend × seasonal combinations (additive, multiplicative, damped); seasonal period configurable |
| `StateSpaceModel` | SSM | Explicit F/G/x matrices; intercept offset; PsiVector for exact h-step variance |
| `ARIMA` `conditionalLeastSquares` | ARIMA CLS | Nonseasonal + seasonal (SARIMA); MA residuals; outlier effects |
| `ARIMA` `exactLeastSquares` | ARIMA Kalman | Companion state-space; Kronecker–Lyapunov P₀; FinalStateVector shortcut |
| `GARCH` | GARCH | ARMAPart (ARMA conditional mean) + GARCHPart (ARCH/GARCH conditional variance) |

**Extra TimeSeriesModel features:**
- `transformation="logarithmic|squareroot"` — applied to training data and inverted on output
- `OutlierEffect` — additive, levelShift, transientChange corrections
- `SeasonalComponent` — SARIMA polynomial convolution φ(B)·Φ(B^m)
- `DynamicRegressor` — transfer function N(B)/D(B); `futureValuesMethod="constant"` or `"userSupplied"`
- `forecast_with_variance(h)` — h-step prediction intervals for all algorithms

## API

### Scoring

```cpp
#include "cPMML.h"

cpmml::Model model("IrisTree.xml");
std::unordered_map<std::string, std::string> sample = {
    {"sepal_length", "6.6"},
    {"sepal_width",  "2.9"},
    {"petal_length", "4.6"},
    {"petal_width",  "1.3"}
};

// Fast raw prediction
std::cout << model.predict(sample);  // "Iris-versicolor"

// Full prediction with probabilities and output fields
cpmml::Prediction pred = model.score(sample);
std::cout << pred.as_string();       // "Iris-versicolor"
for (auto& [cls, prob] : pred.distribution())
    std::cout << cls << ": " << prob << "\n";
```

### Time series forecasting

```cpp
#include "cPMML.h"

cpmml::Model model("AirPassengers_ETS.zip", true);

// Point forecasts
std::vector<double> forecast = model.forecast(12);

// Point forecasts + variance
std::vector<std::pair<double, double>> fv = model.forecast_with_variance(12);
for (auto [point, var] : fv)
    std::cout << point << " ± " << std::sqrt(var) << "\n";

// With user-supplied regressor values
std::unordered_map<std::string, std::vector<double>> regressors = {
    {"gdp", {1.02, 1.03, 1.04}}
};
std::vector<double> forecast_reg = model.forecast(3, regressors);
```

## Set-up

#### Linux / Mac
```
git clone https://github.com/AmadeusITGroup/cPMML.git && cd cPMML && ./install.sh
```
##### Prerequisites
- Git
- CMake >= 3.5.1
- Compiler with C++20 support (GCC 10+, Clang 12+, MSVC 2019+)
- Eigen3 (header-only, included via `CPM`)

#### Windows
```
git clone https://github.com/AmadeusITGroup/cPMML.git && cd cPMML && install.bat
```
##### Prerequisites
- Git
- CMake >= 3.5.1
- MinGW-W64 with C++20 support

## Documentation

Please refer to the [official documentation](https://amadeusitgroup.github.io/cPMML/) for further details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit your pull requests.

## Authors

* **Paolo Iannino** - *Initial work* - [Paolo](https://github.com/piannino)
* **Khalid Daoud** - *Time series models (ETS, SSM, ARIMA, GARCH, DynamicRegressor)*

See also the list of [contributors](https://github.com/AmadeusITGroup/cPMML/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
