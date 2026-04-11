file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(unit_tests ALL)

add_executable(pmml_tester.exe pmml_tester.cc)
target_include_directories(pmml_tester.exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(pmml_tester.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(pmml_tester.exe PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(pmml_tester.exe ${PROJECT_NAME} ${ADDITIONAL_LINK_LIBRARIES})
add_dependencies(unit_tests pmml_tester.exe)

# Auto-discover fixtures: every test/data/model/<name>.zip that has a matching
# test/data/dataset/<name>.csv is registered as a test automatically.
# CONFIGURE_DEPENDS re-triggers cmake when the glob result changes, so new
# fixtures are picked up on the next build without editing this file.
file(GLOB _model_zips CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/data/model/*.zip")

foreach(_zip ${_model_zips})
    get_filename_component(_stem "${_zip}" NAME_WE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/dataset/${_stem}.csv")
        add_test(${_stem} pmml_tester.exe
            "data/model/${_stem}.zip"
            "data/dataset/${_stem}.csv")
    endif()
endforeach()

add_custom_command(
        TARGET unit_tests
        COMMENT "Running unit tests..."
        POST_BUILD
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)
