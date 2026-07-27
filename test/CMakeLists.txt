file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Not part of ALL: a plain `make` builds the tester but does not run the suite.
# Run the fixtures with `ctest --output-on-failure` or `make unit_tests`.
add_custom_target(unit_tests)

add_executable(pmml_tester.exe pmml_tester.cc)
target_include_directories(pmml_tester.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(pmml_tester.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(pmml_tester.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(pmml_tester.exe ${PROJECT_NAME} pugixml::pugixml miniz Eigen3::Eigen ${ADDITIONAL_LINK_LIBRARIES})
add_dependencies(unit_tests pmml_tester.exe)

# Guards the fixture suite cannot express: default-constructed API objects,
# empty containers, and documents that are malformed rather than merely wrong.
add_executable(api_guards.exe api_guards.cc)
target_include_directories(api_guards.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(api_guards.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(api_guards.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(api_guards.exe ${PROJECT_NAME} pugixml::pugixml miniz Eigen3::Eigen ${ADDITIONAL_LINK_LIBRARIES})
add_dependencies(unit_tests api_guards.exe)
add_test(api_guards api_guards.exe)

# Concurrent scoring: a data race does not fail a single-threaded fixture, so
# nothing else in the suite would notice if R3 regressed. Two models, one with
# categorical string inputs (the string-interning path) and one ensemble (the
# OMP segment loop).
add_executable(concurrency.exe concurrency.cc)
target_include_directories(concurrency.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(concurrency.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(concurrency.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(concurrency.exe ${PROJECT_NAME} pugixml::pugixml miniz Eigen3::Eigen ${ADDITIONAL_LINK_LIBRARIES})
find_package(Threads REQUIRED)
target_link_libraries(concurrency.exe Threads::Threads)
add_dependencies(unit_tests concurrency.exe)

add_test(concurrency_tree concurrency.exe
    "data/model/AuditTree.zip" "data/dataset/AuditTree.csv" "Employment")
add_test(concurrency_forest concurrency.exe
    "data/model/AuditRandomForest.zip" "data/dataset/AuditRandomForest.csv" "Occupation")

# Auto-discover fixtures: every test/data/model/<name>.zip that has a matching
# test/data/dataset/<name>.csv is registered as a test automatically.
# CONFIGURE_DEPENDS re-triggers cmake when the glob result changes, so new
# fixtures are picked up on the next build without editing this file.
file(GLOB _model_zips CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/data/model/*.zip")

set(_orphan_models "")
foreach(_zip ${_model_zips})
    get_filename_component(_stem "${_zip}" NAME_WE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/dataset/${_stem}.csv")
        add_test(${_stem} pmml_tester.exe
            "data/model/${_stem}.zip"
            "data/dataset/${_stem}.csv")
    else()
        list(APPEND _orphan_models ${_stem})
    endif()
endforeach()

# A model with no dataset used to drop out of the suite without a word, so a
# fixture could stop running the moment its CSV was renamed and the count would
# still look healthy.
if(_orphan_models)
    message(WARNING "fixture models with no matching dataset, not registered: ${_orphan_models}")
endif()

# Locale-independence guard: numeric parsing and formatting must not go through
# LC_NUMERIC, so these fixtures must produce identical results under a
# comma-decimal locale. Skipped (exit 77) when the host lacks the locale.
foreach(_stem HousingGBTRegressor IrisMultinomReg)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/dataset/${_stem}.csv")
        add_test(${_stem}_comma_locale pmml_tester.exe
            "data/model/${_stem}.zip"
            "data/dataset/${_stem}.csv"
            --locale de_DE.UTF-8)
        set_tests_properties(${_stem}_comma_locale PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endforeach()

add_custom_command(
        TARGET unit_tests
        COMMENT "Running unit tests..."
        POST_BUILD
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)
