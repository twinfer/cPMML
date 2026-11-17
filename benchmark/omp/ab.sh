#!/bin/bash
# Q12: interleaved 4-way run after docs/design/typed-core/aab.sh. $1 holds four
# builds: base-build and head (THREADS=1), base-omp and head-omp (THREADS=4,
# -DOpenMP_ROOT=$(brew --prefix libomp)). A base seq, B head seq,
# C base OpenMP, D head OpenMP; order rotated each round. $2 = rounds (default 15).
D=$1; R=${2:-15}; cd "$(git rev-parse --show-toplevel)/benchmark"
M="AuditBinaryReg:Audit AuditRandomForest:Audit AuditTree:Audit ensemble_audit_dectree:Audit ensemble_iris_dectree:Iris IrisLinearReg:Iris IrisMultinomReg:Iris IrisRandomForest:Iris IrisTree:Iris single_audit_dectree:Audit single_iris_dectree:Iris HousingGBTlassifier:Housing HousingRFClassifier:Housing HousingLRClassifier:Housing HousingGBTlassifier_PCA:Housing HousingRFClassifier_PCA:Housing HousingLRClassifier_PCA:Housing HousingGBTRegressor:Housing HousingRFRegressor:Housing HousingLinearRegressor:Housing HousingGBTRegressor_PCA:Housing HousingRFRegressor_PCA:Housing HousingLinearRegressor_PCA:Housing"
for r in $(seq 1 $R); do
 for md in $M; do m=${md%%:*}; ds=${md##*:}
  case $((r%4)) in 0) order="A B C D";; 1) order="B C D A";; 2) order="C D A B";; 3) order="D A B C";; esac
  for v in $order; do case $v in A) b=base-build;; B) b=head;; C) b=base-omp;; D) b=head-omp;; esac
   echo "$r $v $m $($D/$b/benchmark/model_benchmark.exe data/model/$m.xml data/dataset/$ds.csv)"; done; done; done
