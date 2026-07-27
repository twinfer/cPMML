#!/usr/bin/env python3
"""
scikit-learn oracle for PMML GaussianProcessModel.

This module reads a PMML document -- the same XML cPMML reads -- and answers
"what should this model predict?" using scikit-learn's GaussianProcessRegressor.
It never sees cPMML's parameters or code, so agreement between the two is
evidence and not a tautology.

--------------------------------------------------------------------------
Kernel definitions (PMML 4.4, http://dmg.org/pmml/v4-4/GaussianProcess.html,
cross-checked against pmml-4-4.xsd) and their scikit-learn equivalents.
p is the number of active fields; lambda defaults to 1 throughout.

  RadialBasisKernel            k(x,z) = g * exp(-1/2 * sum_i ((x_i-z_i)/lam)^2)
    attrs: gamma=1, noiseVariance=1, lambda=1   (lambda is a scalar attribute)
    ->  ConstantKernel(g) * RBF(length_scale=lam)

  ARDSquaredExponentialKernel  k(x,z) = g * exp(-1/2 * sum_i ((x_i-z_i)/lam_i)^2)
    attrs: gamma=1, noiseVariance=1;  child Lambda/Array of p reals
    ->  ConstantKernel(g) * RBF(length_scale=[lam_1..lam_p])

  AbsoluteExponentialKernel    k(x,z) = g * exp(-1/2 * sum_i |x_i-z_i|/lam_i)
    attrs: gamma=1, noiseVariance=1;  child Lambda/Array
    ->  ConstantKernel(g) * Matern(length_scale=2*lam, nu=0.5)   [1-D ONLY]

  GeneralizedExponentialKernel k(x,z) = g * exp(-1/2 * sum_i (|x_i-z_i|/lam_i)^degree)
    attrs: gamma=1, noiseVariance=1, degree=1;  child Lambda/Array
    ->  degree == 2 : identical to ARDSquaredExponentialKernel
        degree == 1 : identical to AbsoluteExponentialKernel

Why AbsoluteExponentialKernel is restricted to one dimension: scikit-learn's
Matern(nu=0.5) is exp(-||(x-z)/l||_2), an L2 norm over the scaled difference,
whereas PMML sums |x_i-z_i|/lam_i coordinate-wise (L1).  The two agree only
when p == 1, where ||.||_2 == |.| and l = 2*lam reproduces the PMML factor of
1/2.  scikit-learn has no anisotropic L1 exponential kernel, so multi-input
AbsoluteExponentialKernel is left uncovered rather than covered by a
hand-written kernel that would just restate cPMML's own formula.

Degrees other than 1 and 2 are likewise uncovered.  The two that are covered
pin down the lambda/degree interaction at both ends, and the
GeneralizedExponential == ARD / == AbsoluteExponential identities are asserted
by dedicated fixtures.

--------------------------------------------------------------------------
Predictive equations (identical in PMML and scikit-learn):

    mean(x*) = k*^T (K + noiseVariance*I)^-1 y
    var(x*)  = k(x*,x*) - k*^T (K + noiseVariance*I)^-1 k*

scikit-learn's `alpha` is exactly the value added to the diagonal, so
alpha = noiseVariance.  optimizer=None keeps the hyperparameters at the values
the PMML states, and normalize_y=False keeps y untouched.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

VERSION = f"scikit-learn {sklearn.__version__}"

KERNEL_ELEMENTS = (
    "RadialBasisKernel",
    "ARDSquaredExponentialKernel",
    "AbsoluteExponentialKernel",
    "GeneralizedExponentialKernel",
)


class Unsupported(Exception):
    """The oracle cannot express this model; the fixture must be skipped."""


# ---------------------------------------------------------------------------
# PMML parsing
# ---------------------------------------------------------------------------


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(node: ET.Element, name: str) -> ET.Element | None:
    for child in node:
        if _localname(child.tag) == name:
            return child
    return None


def _findall(node: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in node if _localname(c.tag) == name]


def _descend(root: ET.Element, name: str) -> ET.Element | None:
    for node in root.iter():
        if _localname(node.tag) == name:
            return node
    return None


@dataclass
class GaussianProcess:
    """Everything the oracle needs, read straight out of the PMML."""

    kernel_name: str
    gamma: float
    noise_variance: float
    lam: float | None  # RadialBasisKernel scalar attribute
    lambdas: list[float] | None  # Lambda/Array for the other three
    degree: float
    feature_fields: list[str]
    target_field: str
    x_train: np.ndarray
    y_train: np.ndarray

    def sklearn_kernel(self):
        """The scikit-learn kernel matching this PMML kernel exactly."""
        p = self.x_train.shape[1]
        amplitude = ConstantKernel(self.gamma, constant_value_bounds="fixed")

        def lengths() -> list[float]:
            lams = self.lambdas if self.lambdas is not None else [1.0] * p
            if len(lams) != p:
                raise Unsupported(f"Lambda has {len(lams)} entries but the model has {p} active fields")
            return lams

        def squared_exponential():
            return amplitude * RBF(length_scale=lengths(), length_scale_bounds="fixed")

        def absolute_exponential():
            if p != 1:
                raise Unsupported(
                    "AbsoluteExponentialKernel with more than one active field: PMML sums "
                    "|x_i-z_i|/lambda_i coordinate-wise, scikit-learn's Matern(nu=0.5) uses "
                    "the L2 norm; they coincide only in one dimension"
                )
            # PMML: g*exp(-1/2 * |x-z|/lam);  Matern(nu=0.5): exp(-|x-z|/l).
            # Setting l = 2*lam reproduces the 1/2.
            return amplitude * Matern(length_scale=2.0 * lengths()[0], nu=0.5, length_scale_bounds="fixed")

        if self.kernel_name == "RadialBasisKernel":
            lam = 1.0 if self.lam is None else self.lam
            return amplitude * RBF(length_scale=lam, length_scale_bounds="fixed")

        if self.kernel_name == "ARDSquaredExponentialKernel":
            return squared_exponential()

        if self.kernel_name == "AbsoluteExponentialKernel":
            return absolute_exponential()

        if self.kernel_name == "GeneralizedExponentialKernel":
            if self.degree == 2:
                return squared_exponential()
            if self.degree == 1:
                return absolute_exponential()
            raise Unsupported(
                f"GeneralizedExponentialKernel degree={self.degree}: scikit-learn has no "
                "equivalent; only degree 1 and 2 are oracle-covered"
            )

        raise Unsupported(f"unknown kernel {self.kernel_name}")


def parse(pmml_text: str) -> GaussianProcess:
    root = ET.fromstring(pmml_text)
    model = _descend(root, "GaussianProcessModel")
    if model is None:
        raise Unsupported("document contains no GaussianProcessModel")

    # --- target field -------------------------------------------------------
    schema = _find(model, "MiningSchema")
    if schema is None:
        raise Unsupported("GaussianProcessModel has no MiningSchema")
    target = None
    for f in _findall(schema, "MiningField"):
        if f.get("usageType") in ("target", "predicted"):
            target = f.get("name")
    if target is None:
        raise Unsupported("MiningSchema declares no target field")

    # --- kernel -------------------------------------------------------------
    kernel = next((_find(model, name) for name in KERNEL_ELEMENTS if _find(model, name) is not None), None)
    if kernel is None:
        raise Unsupported("GaussianProcessModel declares no supported kernel")
    kernel_name = _localname(kernel.tag)

    def attr(name: str, default: float) -> float:
        raw = kernel.get(name)
        return default if raw is None else float(raw)

    raw_lambda = kernel.get("lambda")

    lambdas = None
    lam_node = _find(kernel, "Lambda")
    if lam_node is not None:
        array = _find(lam_node, "Array")
        text = (array.text if array is not None else lam_node.text) or ""
        lambdas = [float(v) for v in re.split(r"[\s,]+", text.strip()) if v]

    # --- training instances -------------------------------------------------
    training = _find(model, "TrainingInstances")
    if training is None:
        raise Unsupported("GaussianProcessModel has no TrainingInstances")

    feature_fields: list[str] = []
    feature_columns: list[str] = []
    target_column = None
    fields = _find(training, "InstanceFields")
    if fields is None:
        raise Unsupported("TrainingInstances has no InstanceFields")
    for f in _findall(fields, "InstanceField"):
        field_name = f.get("field")
        if field_name is None:
            raise Unsupported("InstanceField without a field attribute")
        column = f.get("column", field_name)
        if field_name == target:
            target_column = column
        else:
            feature_fields.append(field_name)
            feature_columns.append(column)
    if target_column is None:
        raise Unsupported(f"InstanceFields does not map the target field {target!r}")

    table = _find(training, "InlineTable")
    if table is None:
        raise Unsupported("TrainingInstances has no InlineTable (only inline training data is supported)")

    x_rows, y_rows = [], []
    for row in _findall(table, "row"):
        cells = {_localname(c.tag): (c.text or "").strip() for c in row}
        x_rows.append([float(cells[c]) for c in feature_columns])
        y_rows.append(float(cells[target_column]))

    return GaussianProcess(
        kernel_name=kernel_name,
        gamma=attr("gamma", 1.0),
        noise_variance=attr("noiseVariance", 1.0),
        lam=None if raw_lambda is None else float(raw_lambda),
        lambdas=lambdas,
        degree=attr("degree", 1.0),
        feature_fields=feature_fields,
        target_field=target,
        x_train=np.asarray(x_rows, dtype=float),
        y_train=np.asarray(y_rows, dtype=float),
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def spec_kernel_matrix(gp: GaussianProcess, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The PMML formulas transcribed literally, for use as a self-check.

    This is a direct reading of the spec text, not a route to the predictions:
    predict() goes through scikit-learn.  Comparing the two matrices catches a
    mis-translation in GaussianProcess.sklearn_kernel() -- the one hand-written
    step between the PMML document and the oracle.
    """
    p = a.shape[1]
    lams = np.asarray(gp.lambdas if gp.lambdas is not None else [1.0] * p, dtype=float)
    if gp.kernel_name == "RadialBasisKernel":
        lams = np.full(p, 1.0 if gp.lam is None else gp.lam)

    diff = np.abs(a[:, None, :] - b[None, :, :]) / lams

    if gp.kernel_name in ("RadialBasisKernel", "ARDSquaredExponentialKernel"):
        exponent = (diff**2).sum(axis=2)
    elif gp.kernel_name == "AbsoluteExponentialKernel":
        exponent = diff.sum(axis=2)
    elif gp.kernel_name == "GeneralizedExponentialKernel":
        exponent = (diff**gp.degree).sum(axis=2)
    else:
        raise Unsupported(f"unknown kernel {gp.kernel_name}")

    return gp.gamma * np.exp(-0.5 * exponent)


def check_mapping(gp: GaussianProcess, probe: np.ndarray | None = None) -> None:
    """Assert that the scikit-learn kernel equals the transcribed spec formula."""
    points = gp.x_train if probe is None else probe
    expected = spec_kernel_matrix(gp, points, points)
    actual = gp.sklearn_kernel()(points)
    if not np.allclose(expected, actual, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"{gp.kernel_name}: scikit-learn kernel disagrees with the PMML formula\n"
            f"max abs difference {np.abs(expected - actual).max():.3e}"
        )


def predict(pmml_text: str, test_rows: list[dict]) -> tuple[GaussianProcess, np.ndarray, np.ndarray]:
    """Score test_rows against the PMML model. Returns (model, mean, variance)."""
    gp = parse(pmml_text)
    check_mapping(gp)

    regressor = GaussianProcessRegressor(
        kernel=gp.sklearn_kernel(),
        alpha=gp.noise_variance,
        optimizer=None,
        normalize_y=False,
    )
    regressor.fit(gp.x_train, gp.y_train)

    x_test = np.asarray([[float(row[f]) for f in gp.feature_fields] for row in test_rows], dtype=float)
    mean, std = regressor.predict(x_test, return_std=True)
    return gp, mean, std**2
