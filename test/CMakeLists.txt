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

# The strict lexical reader, XmlNode's typed getters, documents that must not
# load, and invalid input values that end the record (returnInvalid, asIs) --
# the cases a fixture CSV cannot express.
add_executable(strict_parse.exe strict_parse.cc)
target_include_directories(strict_parse.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(strict_parse.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(strict_parse.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(strict_parse.exe ${PROJECT_NAME} pugixml::pugixml miniz Eigen3::Eigen ${ADDITIONAL_LINK_LIBRARIES})
add_dependencies(unit_tests strict_parse.exe)
add_test(strict_parse strict_parse.exe)

# The enum reader: case-sensitive enumerations, no fallback for an unknown
# value, "unsupported" for a PMML value cPMML does not implement.
add_executable(enum_reader.exe enum_reader.cc)
target_include_directories(enum_reader.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(enum_reader.exe ${PROJECT_NAME} ${ADDITIONAL_LINK_LIBRARIES})
add_dependencies(unit_tests enum_reader.exe)
add_test(enum_reader enum_reader.exe)

# The Java-compatible formatter against Double.toString/Float.toString, from
# tables written by tools/java-format/JavaNumbers.java.
add_executable(java_format.exe java_format.cc)
target_include_directories(java_format.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(java_format.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(java_format.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(java_format.exe ${PROJECT_NAME} pugixml::pugixml miniz Eigen3::Eigen ${ADDITIONAL_LINK_LIBRARIES})
add_dependencies(unit_tests java_format.exe)
add_test(java_format java_format.exe)

# The type library of the typed core (src/core/types/, docs/design/typed-core.md
# §2, P1.1): one check per rule. Header-only and not used by the engine yet, so
# the test links nothing.
add_executable(typed_value.exe typed_value.cc)
target_include_directories(typed_value.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_dependencies(unit_tests typed_value.exe)
add_test(typed_value typed_value.exe)

# Must-not-compile cases of the type library (§2.2, §6.4): each is compiled
# at test time with this build's compiler, once with its control (must
# compile) and once as written (must fail with the intended diagnostic, the
# regex below covering clang's and GCC's wording); see
# types_negative/check_negative.cmake. A case that compiles fails its test.
set(_types_negative_flags ${CMAKE_CXX20_STANDARD_COMPILE_OPTION} -Wall -I${PROJECT_SOURCE_DIR}/src)
if(CMAKE_OSX_SYSROOT)
    list(APPEND _types_negative_flags -isysroot ${CMAKE_OSX_SYSROOT})
endif()
foreach(_case_expect
        "missing_alternative|Incomplete, const cpmml::types::String"
        "uncategorised|every Value alternative is in exactly one category"
        "cold_path|private constructor of class 'cpmml::types::cold::ColdPath'|ColdPath::ColdPath\\(\\)' is private")
    string(FIND "${_case_expect}" "|" _bar)
    string(SUBSTRING "${_case_expect}" 0 ${_bar} _case)
    math(EXPR _bar "${_bar} + 1")
    string(SUBSTRING "${_case_expect}" ${_bar} -1 _expect)
    add_test(NAME types_negative_${_case}
        COMMAND ${CMAKE_COMMAND}
            -DCXX=${CMAKE_CXX_COMPILER}
            "-DFLAGS=${_types_negative_flags}"
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/types_negative/${_case}.cc
            "-DEXPECT=${_expect}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/types_negative/check_negative.cmake)
endforeach()

# The binder in report mode (src/bind/, docs/design/typed-core.md §3.7, P1.2):
# its rule cases, then every fixture, benchmark model and staged fixture bound
# and compared with test/data/bind_baseline.txt (--update rewrites it).
# Header-only and not used by the engine, so the test links only pugixml and
# miniz.
add_executable(bind_report.exe bind_report.cc)
target_include_directories(bind_report.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(bind_report.exe pugixml::pugixml miniz)
add_dependencies(unit_tests bind_report.exe)
add_test(NAME bind_report COMMAND bind_report.exe ${PROJECT_SOURCE_DIR})

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

# The cPMML side of the differential fuzzer (tools/fuzz/, docs/design/typed-core.md
# §8). Built on request only (`--target fuzz_runner.exe`); not a test.
add_executable(fuzz_runner.exe EXCLUDE_FROM_ALL fuzz_runner.cc)
target_include_directories(fuzz_runner.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(fuzz_runner.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(fuzz_runner.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(fuzz_runner.exe ${PROJECT_NAME} pugixml::pugixml miniz Eigen3::Eigen ${ADDITIONAL_LINK_LIBRARIES})

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
