cmake_minimum_required(VERSION 3.14)

# CREDITS
set(COPYRIGHT "2019 AMADEUS. All rights reserved.")
set(AUTHOR "Paolo Iannino")

# PROJECT NAME/VERSION
set(PROJECT_NAME cPMML)
set(PROJECT_VERSION_MAJOR 2)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES CXX)

# Only the CPMML_API surface in include/cPMML.h is exported. Without this the
# library exported some six hundred internal symbols from the global namespace
# -- is_missing, single_exp, greater_than, _or, to_double -- each one able to
# collide with, or be interposed by, anything the consumer links. Set before
# the dependencies are declared so pugixml and miniz, which are compiled into
# libcPMML and not re-exported by it, are covered too.
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Emscripten compiles catch handlers away unless exception support is asked
# for, and cPMML relies on catching in places -- InternalScore catches the
# ParsingException from to_double to mean "this score is not a number". Without
# this the wasm build succeeds and then misbehaves at runtime, with an exception
# the library would normally swallow escaping to the caller instead. Not a
# tuning knob. Set before the dependencies are declared: add_compile_options
# only reaches directories added after it, and an exception unwinding through a
# pugixml or miniz frame built without it aborts instead of reaching the catch.
if(EMSCRIPTEN)
    add_compile_options(-fexceptions)
    add_link_options(-fexceptions)
endif()

include(cmake/CPM.cmake)

# DEPENDENCIES

# XML PARSING
# EXCLUDE_FROM_ALL on all three: their install() rules would otherwise run
# during `make install` for cPMML, so installing this library dropped 600 files
# -- the whole of Eigen's header tree -- into the consumer's prefix.
CPMAddPackage(NAME pugixml GITHUB_REPOSITORY zeux/pugixml GIT_TAG v1.14 EXCLUDE_FROM_ALL YES)

# COMPRESSION
CPMAddPackage(NAME miniz
GIT_REPOSITORY https://github.com/richgel999/miniz.git
GIT_TAG 3.1.1
EXCLUDE_FROM_ALL YES
)

# LINEAR ALGEBRA (neural networks)
CPMAddPackage(
    NAME Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
    OPTIONS
        "EIGEN_BUILD_DOC OFF"
        "BUILD_TESTING OFF"
        "EIGEN_BUILD_PKGCONFIG OFF"
        "EIGEN_BUILD_BTL OFF"
    EXCLUDE_FROM_ALL YES
)

# BUILD OPTIONS
# BUILD_TYPE is this project's legacy knob; an explicit -DCMAKE_BUILD_TYPE from
# the caller must still win, otherwise it is silently overwritten below.
if(NOT DEFINED BUILD_TYPE)
    if(CMAKE_BUILD_TYPE)
        set(BUILD_TYPE ${CMAKE_BUILD_TYPE})
    else()
        set(BUILD_TYPE Release)
    endif()
endif()

# No -ffast-math: this is a numerical scoring library whose results are compared
# against a reference implementation (JPMML), and -ffast-math both perturbs those
# results and makes NaN/inf handling undefined.
#
# No -flto here either. In a static archive -flto leaves compiler IR rather than
# machine code in the objects, which only the exact compiler that wrote it can
# link, and CMake archives it with plain ar rather than gcc-ar/llvm-ar. LTO is
# enabled per target below, through CheckIPOSupported, for the shared library
# only, where it is resolved at link time and nothing but machine code ships.
set(RELEASE_FLAGS "-Wall -O3")
set(DEBUG_FLAGS "-Wall -g -Og")
set(VERBOSE OFF)

# SANITIZE=thread|address|undefined|address,undefined. Off by default; the
# concurrency test in particular only proves anything under -fsanitize=thread,
# because a data race passes a plain build almost every time. -Wextra is on
# here rather than in the default flags so that turning it on does not turn
# every warning in the dependencies into noise.
set(SANITIZE "" CACHE STRING "Comma-separated sanitizers to build with (thread, address, undefined)")
if(SANITIZE)
    add_compile_options(-fsanitize=${SANITIZE} -fno-omit-frame-pointer -g -Wextra)
    add_link_options(-fsanitize=${SANITIZE})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE ${BUILD_TYPE})
set(CMAKE_CXX_FLAGS_RELEASE ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "")

if(BUILD_TYPE MATCHES "Debug")
    set(VERBOSE ON)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ${VERBOSE})
set(CMAKE_VERBOSE_MAKEFILE ${VERBOSE})
if (CMAKE_GENERATOR MATCHES "Makefiles")
    set_property(GLOBAL PROPERTY TARGET_MESSAGES ${VERBOSE})
endif ()

set(ADDITIONAL_LINK_LIBRARIES "")
set(CPMML_USES_OPENMP OFF)

# DEBUG OUTPUT (enables field name storage and to_string() in Feature/Sample)
option(CPMML_DEBUG "Enable verbose debug output" OFF)
if(CPMML_DEBUG)
    add_definitions(-DDEBUG)
endif()

# MULTITHREADING
#
# Under Emscripten the default is 1. Emscripten ships an OpenMP runtime, so
# find_package(OpenMP) succeeds there, but that runtime is built on wasm
# threads: it links with shared memory, which needs every object compiled with
# atomics, and a threaded wasm module only runs where SharedArrayBuffer is
# available (cross-origin isolated pages). Asking for THREADS > 1 explicitly
# still works; the dependencies are then compiled with -pthread as well, below.
if(NOT DEFINED THREADS)
    if(EMSCRIPTEN)
        set(THREADS 1)
    else()
        set(THREADS 4)
    endif()
endif()
find_package(OpenMP 4.0)
if(THREADS AND THREADS GREATER 1)
    if (OPENMP_FOUND)# AND OPENMP_VERSION GREATER_EQUAL 4)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        add_definitions(-DMULTITHREADING)
        add_definitions(-DNUM_THREADS=${THREADS})
        # Link the runtime too, not only the compile flags: a static archive
        # passes it on to the consumer (cPMMLConfig.cmake finds it there), and
        # a shared object built without it leaves the GOMP/omp symbols for the
        # executable to resolve.
        set(CPMML_USES_OPENMP ON)
        # The OpenMP flags above reach only targets declared from here on.
        # pugixml and miniz were declared earlier, and under Emscripten an
        # object compiled without atomics makes wasm-ld refuse the
        # shared-memory link that the threaded runtime needs.
        if(EMSCRIPTEN)
            get_target_property(CPMML_PUGIXML_IMPL pugixml INTERFACE_LINK_LIBRARIES)
            target_compile_options(${CPMML_PUGIXML_IMPL} PRIVATE -pthread)
            target_compile_options(miniz PRIVATE -pthread)
        endif()
    else()
        message("OpenMP is needed to enable multithreading.")
    endif()
endif ()

# BUILD LIBRARY
include(GNUInstallDirs)

add_definitions(-DCPMML_VERSION=\"${PROJECT_VERSION}\")
# SHARED was hardcoded. A static archive is the normal thing to want for a
# WebAssembly target -- Emscripten turns a SHARED library into a side module
# needing -fPIC throughout, which the vendored miniz is not built with -- and it
# is what anyone embedding the scorer in a single binary wants too.
#
# Deliberately not CMake's BUILD_SHARED_LIBS: that variable is read by pugixml,
# miniz and Eigen as well, so setting it would also flip *them* to shared, which
# turns three vendored build-tree targets into runtime dependencies a consumer
# has to find. The dependencies stay static and get absorbed either way; this
# switch is about cPMML alone.
#
# Defaults to static under Emscripten, where shared is the case that fails.
if(EMSCRIPTEN)
    set(CPMML_SHARED_DEFAULT OFF)
else()
    set(CPMML_SHARED_DEFAULT ON)
endif()
option(CPMML_SHARED "Build cPMML as a shared library rather than a static archive" ${CPMML_SHARED_DEFAULT})
if(CPMML_SHARED)
    set(CPMML_LIBRARY_TYPE SHARED)
else()
    set(CPMML_LIBRARY_TYPE STATIC)
endif()

add_library(
        ${PROJECT_NAME} ${CPMML_LIBRARY_TYPE}
        src/core/header.h
        src/api/exceptions.cc
        src/api/model.cc
        src/api/prediction.cc
        src/api/version.cc
        src/core/xmlnode.h
        src/utils/csvreader.h
        src/utils/utils.h
        src/core/property.h
        src/core/intervalbuilder.h
        src/core/miningschema.h
        src/core/fastpredicate.h
        src/core/miningfunction.h
        src/core/sample.h
        src/core/transformationdictionary.h
        src/core/datatype.h
        src/core/builtinfunctions.h
        src/core/invalidvaluetreatmentmethod.h
        src/core/outliertreatmentmethod.h
        src/core/derivedfield.h
        src/core/datafield.h
        src/core/closure.h
        src/core/value.h
        src/core/internal_score.h
        src/core/fieldusagetype.h
        src/core/indexer.h
        src/core/predicateoptype.h
        src/core/dagbuilder.h
        src/core/predicatetype.h
        src/core/missingvaluetreatmentmethod.h
        src/core/internal_model.h
        src/core/datadictionary.h
        src/core/modelbuilder.h
        src/core/predicate.h
        src/core/optype.h
        src/core/internal_evaluator.h
        src/core/model_evaluator.h
        src/core/enum_parser.h
        src/core/traininginstances.h
        src/core/miningfield.h
        src/core/target.h
        src/treemodel/treescore.h
        src/core/predicatebuilder.h
        src/treemodel/scoredistribution.h
        src/treemodel/node.h
        src/treemodel/treemodel.h
        src/regressionmodel/regressionscore.h
        src/regressionmodel/regressiontable.h
        src/regressionmodel/numericpredictor.h
        src/regressionmodel/regressionmodel.h
        src/regressionmodel/predictorterm.h
        src/regressionmodel/categoricalpredictor.h
        src/regressionmodel/normalizationmethodtype.h
        src/regressionmodel/normalizationmethodbuilder.h
        src/ensemblemodel/ensemblemodel.h
        src/ensemblemodel/segment.h
        src/ensemblemodel/multiplemodelmethod.h
        src/math/misc.h
        src/math/normalizationmethods.h
        src/treetable/treetablenode.h
        src/treetable/treetable.h
        src/expression/expression.h
        src/expression/expressiontype.h
        src/expression/discretize.h
        src/expression/mapvalues.h
        src/expression/normdiscrete.h
        src/expression/normcontinuous.h
        src/expression/constant.h
        src/expression/expressionbuilder.h
        src/expression/apply.h
        src/expression/fieldref.h
        src/expression/textindex.h
        src/output/outputexpression.h
        src/output/outputexpressiontype.h
        src/output/outputexpressionbuilder.h
        src/output/outputfield.h
        src/output/outputdictionary.h
        src/output/predictedValue.h
        src/output/probability.h
        src/output/transformedvalue.h
        src/neuralnetwork/neuralactivation.h
        src/neuralnetwork/neuralnetworkmodel.h
        src/svmmodel/svmkernel.h
        src/svmmodel/svmmodel.h
        src/knnmodel/knnmodel.h
        src/generalregressionmodel/generalregressionmodel.h
        src/scorecard/scorecardmodel.h
        src/naivebayesmodel/naivebayesmodel.h
        src/clusteringmodel/clusteringmodel.h
        src/rulesetmodel/rulesetmodel.h
        src/baselinemodel/baselinemodel.h
        src/anomalydetectionmodel/anomalydetectionevaluator.h
        src/gaussianprocessmodel/gaussianprocessmodel.h
        src/textmodel/textmodel.h
        src/timeseriesmodel/timeseriesmodel.h
        src/timeseriesmodel/timeseriesevaluator.h
        third_party/boost/integral.h)

# INSTALL LIBRARY
set_target_properties(
        ${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        # SONAME carries the major version alone. It was the full 2.1, which
        # defeats the point: every minor release minted a new SONAME and forced
        # a relink even when nothing in the ABI had changed.
        SOVERSION ${PROJECT_VERSION_MAJOR}
        LINKER_LANGUAGE CXX
        PUBLIC_HEADER include/cPMML.h
)

target_include_directories(
        ${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PRIVATE third_party)

# PRIVATE, not PUBLIC. cPMML.h includes nothing but standard headers, so none
# of these three is part of the interface -- linking them PUBLIC put pugixml's,
# miniz's and Eigen's include directories and link requirements onto every
# consumer, and made an Eigen version mismatch in the consumer's own build a
# cPMML problem.
#
# BUILD_INTERFACE because install(EXPORT) refuses to export a static archive
# whose link interface names targets that exist only in this build tree. It
# changes nothing for the shared library, whose private dependencies are not
# part of the exported interface in the first place.
target_link_libraries(
        ${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:pugixml::pugixml>
        $<BUILD_INTERFACE:miniz>
        $<BUILD_INTERFACE:Eigen3::Eigen>
        ${ADDITIONAL_LINK_LIBRARIES}
)
if(CPMML_USES_OPENMP)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CPMML_SHARED)
    # The shared object absorbs the static pugixml and miniz archives at link
    # time, with their symbols hidden, so it is self-contained already.
    include(CheckIPOSupported)
    check_ipo_supported(RESULT CPMML_IPO_SUPPORTED OUTPUT CPMML_IPO_OUTPUT LANGUAGES CXX)
    if(CPMML_IPO_SUPPORTED)
        set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(STATUS "cPMML: LTO not supported, building without it: ${CPMML_IPO_OUTPUT}")
    endif()
else()
    # A static archive absorbs nothing: left alone, the installed libcPMML.a
    # would need pugixml's and miniz's archives, which are build-tree targets
    # and are never installed. Fold their object files into it instead, so the
    # installed archive is self-contained just as the shared object is. Eigen
    # is header-only and needs nothing. pugixml::pugixml is an INTERFACE target
    # in front of the real library, which is resolved here.
    if(CMAKE_VERSION VERSION_LESS 3.15)
        message(FATAL_ERROR "CPMML_SHARED=OFF needs CMake 3.15 or newer ($<TARGET_OBJECTS> on a static library)")
    endif()
    get_target_property(CPMML_PUGIXML_TARGET pugixml INTERFACE_LINK_LIBRARIES)
    target_sources(
            ${PROJECT_NAME} PRIVATE
            $<TARGET_OBJECTS:${CPMML_PUGIXML_TARGET}>
            $<TARGET_OBJECTS:miniz>
    )
endif()

include(CMakePackageConfigHelpers)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Consumers had no way to say find_package(cPMML) -- the install dropped a
# library and a header and nothing that describes them. pugixml and miniz are
# built into the library either way -- linked into the shared object, or folded
# into the static archive as object files -- and Eigen is header-only, so the
# config needs no find_dependency for any of them. The one exception is the
# OpenMP runtime, which a static archive cannot carry: cPMMLConfig.cmake finds
# it when CPMML_FIND_OPENMP says the archive was built against it.
if(CPMML_USES_OPENMP AND NOT CPMML_SHARED)
    set(CPMML_FIND_OPENMP ON)
else()
    set(CPMML_FIND_OPENMP OFF)
endif()
install(
        EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

configure_package_config_file(
        cmake/${PROJECT_NAME}Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(
        FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# BENCHMARK
add_subdirectory(benchmark)

# TEST
enable_testing()
add_subdirectory(test)

# DOCUMENTATION
add_subdirectory(docsrc)
