file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(unit_tests ALL)

add_executable(pmml_tester.exe pmml_tester.cc)
target_include_directories(pmml_tester.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(pmml_tester.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(pmml_tester.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(pmml_tester.exe ${PROJECT_NAME} ${ADDITIONAL_LINK_LIBRARIES})
add_dependencies(unit_tests pmml_tester.exe)

macro(add_pmml_test TARGET)
    add_test(${TARGET} pmml_tester.exe data/model/${TARGET}.zip data/dataset/${TARGET}.csv)
endmacro()

# Classification / regression models
#add_pmml_test(AuditBinaryReg) # original prediction not available
add_pmml_test(AuditRandomForest)
add_pmml_test(AuditTree)
add_pmml_test(ensemble_audit_dectree)
add_pmml_test(ensemble_iris_dectree)
add_pmml_test(IrisLinearReg)
add_pmml_test(IrisMultinomReg)
add_pmml_test(IrisRandomForest)
add_pmml_test(IrisTree)
add_pmml_test(single_audit_dectree)
add_pmml_test(single_iris_dectree)

add_pmml_test(HousingGBTClassifier)
add_pmml_test(HousingRFClassifier)
add_pmml_test(HousingLRClassifier)
add_pmml_test(HousingLinearRegressor)

add_pmml_test(HousingGBTClassifier_PCA)
add_pmml_test(HousingRFClassifier_PCA)
add_pmml_test(HousingLRClassifier_PCA)
add_pmml_test(HousingGBTRegressor_PCA)
add_pmml_test(HousingRFRegressor_PCA)
add_pmml_test(HousingLinearRegressor_PCA)

add_pmml_test(GRM_Linear)
add_pmml_test(GRM_MultinomLogistic)
add_pmml_test(IrisNN)
add_pmml_test(IrisSVM)
add_pmml_test(IrisKNN)
add_pmml_test(IrisScorecard)
add_pmml_test(IrisNaiveBayes)
add_pmml_test(IrisClustering)
add_pmml_test(IrisRuleSet)
add_pmml_test(IrisBaseline)
add_pmml_test(IrisAnomaly)
add_pmml_test(IrisGP)
add_pmml_test(IrisText)
add_pmml_test(IrisTextIndex)

# Time series models (same macro — mode detected from CSV)
add_pmml_test(ETS_AddTrend)
add_pmml_test(ETS_SeasonalAddAdd)
add_pmml_test(SSM_LinearTrend)
add_pmml_test(SSM_Intercept)
add_pmml_test(SSM_PsiVector)
add_pmml_test(ARIMA_AR1)
add_pmml_test(ARIMA_Kalman)
add_pmml_test(ARIMA_SARIMA)
add_pmml_test(GARCH_AR1_GARCH11)
add_pmml_test(ARIMA_DynReg)

add_custom_command(
        TARGET unit_tests
        COMMENT "Running unit tests..."
        POST_BUILD
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)
