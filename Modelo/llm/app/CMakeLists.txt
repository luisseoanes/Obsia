add_executable(obgyn_chat 
    main.cpp
    bm25.cpp
    rag_engine.cpp
    clinical_engine.cpp
)
target_link_libraries(obgyn_chat PRIVATE llama)
target_compile_features(obgyn_chat PRIVATE cxx_std_17)

# Optimización solo para nuestro código (solo en Release, evita choque con /RTC1 en Debug)
if(MSVC)
    target_compile_options(obgyn_chat PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(obgyn_chat PRIVATE $<$<CONFIG:Release>:-O2>)
endif()
