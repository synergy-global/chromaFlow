# Tests CMakeLists.txt for ChromaFlow

if(CHROMAFLOW_USE_CATCH2)
    # Catch2 tests
    add_executable(chromaflow_tests
        test_chroma_base_classes.cpp
        test_chroma_utils.cpp
    )
    
    target_link_libraries(chromaflow_tests 
        PRIVATE 
        ChromaFlow
        Catch2::Catch2WithMain
    )
    
    target_include_directories(chromaflow_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../chromaLib
    )
    
    # Register tests with CTest
    include(CTest)
    include(Catch)
    catch_discover_tests(chromaflow_tests)
    
else()
    # Google Test tests (alternative)
    add_executable(chromaflow_tests
        test_chroma_base_classes_gtest.cpp
        test_chroma_utils_gtest.cpp
    )
    
    target_link_libraries(chromaflow_tests 
        PRIVATE 
        ChromaFlow
        gtest_main
    )
    
    target_include_directories(chromaflow_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../chromaLib
    )
    
    # Register tests with CTest
    include(GoogleTest)
    gtest_discover_tests(chromaflow_tests)
endif()

# Set C++ standard for tests
target_compile_features(chromaflow_tests PRIVATE cxx_std_17)