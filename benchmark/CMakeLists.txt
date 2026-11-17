file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Not part of ALL. `make benchmark` runs the 11 original benchmarks;
# `make benchmark_all` runs those, the 12 Housing models and one fixture per
# model family (docs/design/typed-core.md §6.4 item 7).
#
# The two lists below are also the model list of tools/bench-ab.sh, which reads
# the add_model_benchmark / add_fixture_benchmark lines of this file: one entry
# per line, arguments separated by spaces. NOISY marks a model whose A/A 95%
# interval is wider than ±5%; the gate gives it 31 rounds instead of 15. For
# the 23 Audit/Iris/Housing models that is the list of §6.4; for the fixtures it
# is those wider than ±5% in at least two of the four groups (sequential and
# OpenMP builds, eval and --read-all) of the 15-round A/A run made with P0.1.
add_custom_target(benchmark)
add_custom_target(benchmark_all)

add_executable(model_benchmark.exe model_benchmark.cc)
target_include_directories(model_benchmark.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(model_benchmark.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(model_benchmark.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(model_benchmark.exe ${PROJECT_NAME} pugixml::pugixml miniz Eigen3::Eigen ${ADDITIONAL_LINK_LIBRARIES})

# add_model_benchmark(NAME DATASET [NOISY] [ALL_ONLY]): data/model/NAME.xml
# scored once over data/dataset/DATASET.csv. ALL_ONLY keeps it out of
# `benchmark`.
function(add_model_benchmark TARGET DATASET)
    add_custom_target(${TARGET}_benchmark_run
            COMMAND model_benchmark.exe data/model/${TARGET}.xml data/dataset/${DATASET}.csv
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Benchmarking ${TARGET}..."
            )
    if(NOT "ALL_ONLY" IN_LIST ARGN)
        add_dependencies(benchmark ${TARGET}_benchmark_run)
    endif()
    add_dependencies(benchmark_all ${TARGET}_benchmark_run)
endfunction()

# add_fixture_benchmark(NAME REPEAT [NOISY]): the test fixture
# test/data/model/NAME.zip scored REPEAT times over test/data/dataset/NAME.csv.
function(add_fixture_benchmark TARGET REPEAT)
    add_custom_target(${TARGET}_benchmark_run
            COMMAND model_benchmark.exe --zip --repeat ${REPEAT}
                    ${PROJECT_SOURCE_DIR}/test/data/model/${TARGET}.zip
                    ${PROJECT_SOURCE_DIR}/test/data/dataset/${TARGET}.csv
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Benchmarking ${TARGET}..."
            )
    add_dependencies(benchmark_all ${TARGET}_benchmark_run)
endfunction()

add_model_benchmark(AuditBinaryReg Audit)
add_model_benchmark(AuditRandomForest Audit NOISY)
add_model_benchmark(AuditTree Audit)
add_model_benchmark(ensemble_audit_dectree Audit)
add_model_benchmark(ensemble_iris_dectree Iris)
add_model_benchmark(IrisLinearReg Iris NOISY)
add_model_benchmark(IrisMultinomReg Iris NOISY)
add_model_benchmark(IrisRandomForest Iris)
add_model_benchmark(IrisTree Iris)
add_model_benchmark(single_audit_dectree Audit)
add_model_benchmark(single_iris_dectree Iris NOISY)

add_model_benchmark(HousingGBTlassifier Housing ALL_ONLY)
add_model_benchmark(HousingRFClassifier Housing ALL_ONLY)
add_model_benchmark(HousingLRClassifier Housing ALL_ONLY)
add_model_benchmark(HousingGBTlassifier_PCA Housing ALL_ONLY)
add_model_benchmark(HousingRFClassifier_PCA Housing ALL_ONLY)
add_model_benchmark(HousingLRClassifier_PCA Housing ALL_ONLY)
add_model_benchmark(HousingGBTRegressor Housing ALL_ONLY)
add_model_benchmark(HousingRFRegressor Housing ALL_ONLY)
add_model_benchmark(HousingLinearRegressor Housing ALL_ONLY)
add_model_benchmark(HousingGBTRegressor_PCA Housing ALL_ONLY)
add_model_benchmark(HousingRFRegressor_PCA Housing NOISY ALL_ONLY)
add_model_benchmark(HousingLinearRegressor_PCA Housing ALL_ONLY)

# One model per family; REPEAT brings each to roughly 10 000 evaluate calls.
add_fixture_benchmark(fix_nn_missing_classification 700 NOISY)
add_fixture_benchmark(fix_svm_ovo_tie 140 NOISY)
add_fixture_benchmark(fix_minkowski_knn_average_p05 600)
add_fixture_benchmark(fix_clustering_equal_boolean_center 85)
add_fixture_benchmark(fix_nb_threshold 1000 NOISY)
add_fixture_benchmark(GRM_OrdinalCauchit 800)
add_fixture_benchmark(IrisScorecard 1000)
add_fixture_benchmark(IrisRuleSet 330 NOISY)
add_fixture_benchmark(IrisText 2000 NOISY)
add_fixture_benchmark(fix_gp_missing_continue 1100)
add_fixture_benchmark(IrisBaseline 1250)
add_fixture_benchmark(fix_anomaly_miningschema_categorical 650)
add_fixture_benchmark(fix_association_output_defaults 1000)
