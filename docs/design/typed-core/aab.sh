#!/bin/bash
# A/A/C interleaved: A and B are the same sequential binary, C is the OpenMP build.
D=$1; cd "$(git rev-parse --show-toplevel)/benchmark"
M="AuditBinaryReg:Audit AuditRandomForest:Audit AuditTree:Audit ensemble_audit_dectree:Audit ensemble_iris_dectree:Iris IrisLinearReg:Iris IrisMultinomReg:Iris IrisRandomForest:Iris IrisTree:Iris single_audit_dectree:Audit single_iris_dectree:Iris HousingGBTlassifier:Housing HousingRFClassifier:Housing HousingLRClassifier:Housing HousingGBTlassifier_PCA:Housing HousingRFClassifier_PCA:Housing HousingLRClassifier_PCA:Housing HousingGBTRegressor:Housing HousingRFRegressor:Housing HousingLinearRegressor:Housing HousingGBTRegressor_PCA:Housing HousingRFRegressor_PCA:Housing HousingLinearRegressor_PCA:Housing"
for r in $(seq 1 15); do
 for md in $M; do m=${md%%:*}; ds=${md##*:}
  order="A B C"; case $((r%3)) in 1) order="B C A";; 2) order="C A B";; esac
  for v in $order; do b=$D/build-seq; [ $v = C ] && b=$D/build-omp
   echo "$r $v $m $($b/benchmark/model_benchmark.exe data/model/$m.xml data/dataset/$ds.csv)"; done; done; done
