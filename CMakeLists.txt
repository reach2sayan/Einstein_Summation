cmake_minimum_required(VERSION 3.8)
project("Einstein_Summation")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
        # For using std::mdspan with Clang
        add_compile_options(-stdlib=libc++)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wall -Wextra -pedantic)
    endif()
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

add_library(einsum INTERFACE
        ${CMAKE_SOURCE_DIR}/include/fixed_string.hpp
        ${CMAKE_SOURCE_DIR}/include/helper.hpp
        ${CMAKE_SOURCE_DIR}/include/traits.hpp
        ${CMAKE_SOURCE_DIR}/include/einsum.hpp
)

add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC einsum)
target_include_directories(main PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${mdspan_SOURCE_DIR}/include
)

add_executable(tests tests.cpp)
target_link_libraries(tests PUBLIC gtest_main einsum)
target_include_directories(tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${mdspan_SOURCE_DIR}/include
)

add_test(NAME tests COMMAND tests)
enable_testing()
