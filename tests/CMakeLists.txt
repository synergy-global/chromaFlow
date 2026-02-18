if(CHROMAFLOW_USE_CATCH2)
    add_executable(chromaflow_tests
        test_chroma_base_classes.cpp
        test_layers_outputs.cpp
        test_feature_extractor_outputs.cpp
        
    )
    
    target_link_libraries(chromaflow_tests 
        PRIVATE 
        ChromaFlow
        Catch2::Catch2WithMain
    )
    
    target_include_directories(chromaflow_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../chromaLib
    )
    
    include(Catch)
    catch_discover_tests(chromaflow_tests)
    
else()
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
endif()
