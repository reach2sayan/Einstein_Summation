cmake_minimum_required(VERSION 4.0)
project(Einstein_Summation2)
if (MSVC)
    if (POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,
                                                $<CXX_COMPILER_ID:MSVC>>,
                                                $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,
                                                $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else ()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wall -Wextra -pedantic)
        add_compile_options(-stdlib=libc++)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wall -Wextra -pedantic)
    endif()
endif ()
include(FetchContent)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Boost REQUIRED)

FetchContent_Declare(
        mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG stable
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(mdspan)

FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

add_library(einsum INTERFACE
        ${CMAKE_SOURCE_DIR}/include/einsum.hpp
        ${CMAKE_SOURCE_DIR}/include/labels.hpp
        ${CMAKE_SOURCE_DIR}/include/matrices.hpp
        ${CMAKE_SOURCE_DIR}/include/input_handler.hpp
)

add_executable(main main.cpp)
target_link_libraries(main fmt::fmt einsum)
target_include_directories(main PRIVATE
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${mdspan_SOURCE_DIR}/include)

add_executable(tests tests/tests.cpp)
target_link_libraries(tests PUBLIC einsum)
target_include_directories(tests PRIVATE
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${mdspan_SOURCE_DIR}/include
)

add_test(NAME tests COMMAND tests)
enable_testing()
