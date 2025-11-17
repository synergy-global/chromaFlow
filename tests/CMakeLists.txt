# Tests CMakeLists.txt for ChromaFlow

if(CHROMAFLOW_USE_CATCH2)
    # Catch2 tests
    add_executable(chromaflow_tests
        test_chroma_base_classes.cpp
    )
    
    target_link_libraries(chromaflow_tests 
        PRIVATE 
        ChromaFlow
        Catch2::Catch2WithMain
    )
    
    target_include_directories(chromaflow_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../chromaLib
    )
    
    # No CTest integration; run the Catch2 binary directly
    
else()
    # Google Test tests (alternative)
    add_executable(chromaflow_tests
        test_chroma_base_classes_gtest.cpp
    )
    
    target_link_libraries(chromaflow_tests 
        PRIVATE 
        ChromaFlow
        gtest_main
    )
    
    target_include_directories(chromaflow_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../chromaLib
    )
    
    # No CTest integration; run the test binary directly
endif()

# Set C++ standard for tests
target_compile_features(chromaflow_tests PRIVATE cxx_std_17)
