cmake_minimum_required(VERSION 3.8)
project("Einstein_Summation")

set(CMAKE_CXX_STANDARD 26)
if (MSVC)
    if (POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else ()
    add_compile_options(-Wall -Wpedantic -Wfatal-errors)
endif ()

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)
find_package(Eigen3 REQUIRED)
link_directories(${Eigen_INCLUDE_DIRS})

add_executable(einsum main.cpp einsum.cpp)

target_include_directories(einsum PRIVATE
        ${mdspan_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(einsum PUBLIC ${Eigen_LIBRARIES})


#add_executable(tests tests.cpp)
#target_link_libraries(tests gtest_main)
#target_include_directories(tests PRIVATE
#${CMAKE_SOURCE_DIR}/include
#${mdspan_SOURCE_DIR}/include
#)

#add_test(NAME tests COMMAND tests)
#enable_testing()