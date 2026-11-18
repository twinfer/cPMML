# One must-not-compile case of test/types_negative/ (docs/design/typed-core.md
# §2.2, §6.4), run by ctest:
#
#   cmake -DCXX=<compiler> -DFLAGS=<;-list> -DSOURCE=<case.cc> -DEXPECT=<regex> -P check_negative.cmake
#
# 1. The case compiled with -DCPMML_NEGATIVE_CONTROL (the one offending
#    construct removed) must compile: otherwise the case could fail for any
#    reason at all -- a typo, a moved header -- and still "pass".
# 2. The case as written must NOT compile, and the compiler's diagnostics must
#    match EXPECT, the intended reason. A case that compiles fails the test.
foreach(var CXX FLAGS SOURCE EXPECT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "check_negative.cmake: ${var} is not set")
  endif()
endforeach()

execute_process(
  COMMAND ${CXX} ${FLAGS} -fsyntax-only -DCPMML_NEGATIVE_CONTROL ${SOURCE}
  RESULT_VARIABLE control_result
  OUTPUT_VARIABLE control_output
  ERROR_VARIABLE control_output)
if(NOT control_result EQUAL 0)
  message(FATAL_ERROR "the control of ${SOURCE} does not compile, so the case proves nothing:\n${control_output}")
endif()

execute_process(
  COMMAND ${CXX} ${FLAGS} -fsyntax-only ${SOURCE}
  RESULT_VARIABLE case_result
  OUTPUT_VARIABLE case_output
  ERROR_VARIABLE case_output)
if(case_result EQUAL 0)
  message(FATAL_ERROR "${SOURCE} compiled; it must not")
endif()
if(NOT case_output MATCHES "${EXPECT}")
  message(FATAL_ERROR "${SOURCE} failed to compile, but not for the intended reason (expected /${EXPECT}/):\n${case_output}")
endif()
string(REGEX MATCH "[^\n]*error[^\n]*" first_error "${case_output}")
message(STATUS "rejected as intended: ${first_error}")
