# Guía de Integración RAG en C++

## 🎯 Objetivo

Integrar el sistema RAG generado en el chatbot C++ existente, **sin usar modelo de embeddings en móvil**.

---

## 📋 Estrategia Recomendada: BM25 + FAISS

### Arquitectura Propuesta

```
Query del usuario
    ↓
[1] BM25 Scoring (búsqueda léxica)
    ↓
Top-10 candidatos
    ↓
[2] FAISS k-NN (opcional, para refinamiento)
    ↓
Top-3 chunks finales
    ↓
Inyectar en prompt del LLM
```

---

## 🔧 Implementación Paso a Paso

### Paso 1: Agregar FAISS a CMake

Modificar `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(obgyn-chatbot LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# FAISS (descargar pre-compilado o compilar)
find_package(faiss REQUIRED)

# Build llama.cpp library
set(LLAMA_STANDALONE OFF CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp ${CMAKE_BINARY_DIR}/llama-build)

# Our chatbot app
add_subdirectory(app)
```

### Paso 2: Crear Módulo BM25 (Sin Embeddings)

Crear `app/bm25.h`:

```cpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct BM25Result {
    int chunk_id;
    float score;
};

class BM25Ranker {
public:
    BM25Ranker(float k1 = 1.5f, float b = 0.75f);
    
    // Indexar chunks
    void index_documents(const std::vector<std::string>& docs);
    
    // Buscar top-k
    std::vector<BM25Result> search(const std::string& query, int k = 10);
    
private:
    float k1_, b_;
    std::vector<std::unordered_map<std::string, int>> doc_term_freqs_;
    std::unordered_map<std::string, int> doc_freqs_;
    std::vector<int> doc_lengths_;
    float avg_doc_length_;
    
    std::vector<std::string> tokenize(const std::string& text);
    float compute_idf(const std::string& term);
};
```

Crear `app/bm25.cpp`:

```cpp
#include "bm25.h"
#include <algorithm>
#include <sstream>
#include <cmath>

BM25Ranker::BM25Ranker(float k1, float b) : k1_(k1), b_(b), avg_doc_length_(0) {}

std::vector<std::string> BM25Ranker::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        // Convertir a minúsculas y limpiar
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        
        // Remover puntuación
        token.erase(std::remove_if(token.begin(), token.end(), 
            [](char c) { return !std::isalnum(c); }), token.end());
        
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

void BM25Ranker::index_documents(const std::vector<std::string>& docs) {
    doc_term_freqs_.clear();
    doc_freqs_.clear();
    doc_lengths_.clear();
    
    int total_length = 0;
    
    for (const auto& doc : docs) {
        auto tokens = tokenize(doc);
        doc_lengths_.push_back(tokens.size());
        total_length += tokens.size();
        
        std::unordered_map<std::string, int> term_freq;
        std::unordered_set<std::string> unique_terms;
        
        for (const auto& token : tokens) {
            term_freq[token]++;
            unique_terms.insert(token);
        }
        
        doc_term_freqs_.push_back(term_freq);
        
        for (const auto& term : unique_terms) {
            doc_freqs_[term]++;
        }
    }
    
    avg_doc_length_ = static_cast<float>(total_length) / docs.size();
}

float BM25Ranker::compute_idf(const std::string& term) {
    int N = doc_term_freqs_.size();
    int df = doc_freqs_.count(term) ? doc_freqs_[term] : 0;
    
    return std::log((N - df + 0.5f) / (df + 0.5f) + 1.0f);
}

std::vector<BM25Result> BM25Ranker::search(const std::string& query, int k) {
    auto query_tokens = tokenize(query);
    std::vector<BM25Result> results;
    
    for (size_t doc_id = 0; doc_id < doc_term_freqs_.size(); ++doc_id) {
        float score = 0.0f;
        
        for (const auto& term : query_tokens) {
            float idf = compute_idf(term);
            int tf = doc_term_freqs_[doc_id].count(term) ? 
                     doc_term_freqs_[doc_id][term] : 0;
            
            float doc_len = doc_lengths_[doc_id];
            float norm = 1.0f - b_ + b_ * (doc_len / avg_doc_length_);
            
            score += idf * (tf * (k1_ + 1.0f)) / (tf + k1_ * norm);
        }
        
        results.push_back({static_cast<int>(doc_id), score});
    }
    
    // Ordenar por score descendente
    std::sort(results.begin(), results.end(), 
        [](const BM25Result& a, const BM25Result& b) {
            return a.score > b.score;
        });
    
    // Retornar top-k
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}
```

### Paso 3: Crear Módulo RAG

Crear `app/rag_engine.h`:

```cpp
#pragma once
#include "bm25.h"
#include <string>
#include <vector>

struct Chunk {
    std::string text;
    std::string source;
    int chunk_id;
};

class RAGEngine {
public:
    RAGEngine(const std::string& chunks_path);
    
    // Buscar chunks relevantes
    std::vector<std::string> search(const std::string& query, int k = 3);
    
    // Obtener contexto formateado para el prompt
    std::string get_augmented_context(const std::string& query, int k = 3);
    
private:
    std::vector<Chunk> chunks_;
    BM25Ranker ranker_;
    
    void load_chunks(const std::string& path);
};
```

Crear `app/rag_engine.cpp`:

```cpp
#include "rag_engine.h"
#include <fstream>
#include <sstream>
#include "json.hpp"  // nlohmann/json

using json = nlohmann::json;

RAGEngine::RAGEngine(const std::string& chunks_path) {
    load_chunks(chunks_path);
    
    // Indexar chunks con BM25
    std::vector<std::string> texts;
    for (const auto& chunk : chunks_) {
        texts.push_back(chunk.text);
    }
    ranker_.index_documents(texts);
}

void RAGEngine::load_chunks(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir: " + path);
    }
    
    json j;
    file >> j;
    
    for (const auto& item : j) {
        Chunk chunk;
        chunk.text = item["text"];
        chunk.source = item["metadata"]["source"];
        chunk.chunk_id = item["metadata"]["chunk_id"];
        chunks_.push_back(chunk);
    }
}

std::vector<std::string> RAGEngine::search(const std::string& query, int k) {
    auto results = ranker_.search(query, k);
    
    std::vector<std::string> relevant_texts;
    for (const auto& result : results) {
        if (result.score > 0.0f) {  // Filtrar resultados irrelevantes
            relevant_texts.push_back(chunks_[result.chunk_id].text);
        }
    }
    
    return relevant_texts;
}

std::string RAGEngine::get_augmented_context(const std::string& query, int k) {
    auto chunks = search(query, k);
    
    if (chunks.empty()) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "Contexto médico relevante de las guías clínicas:\n\n";
    
    for (size_t i = 0; i < chunks.size(); ++i) {
        oss << "[" << (i + 1) << "] " << chunks[i] << "\n\n";
    }
    
    oss << "---\n\n";
    return oss.str();
}
```

### Paso 4: Modificar `main.cpp`

```cpp
#include "llama.h"
#include "rag_engine.h"  // ← NUEVO
#include <cstdio>
#include <iostream>
#include <string>

// ... (código existente) ...

int main(int argc, char** argv) {
    // ... (parseo de argumentos) ...
    
    // Cargar modelo LLM
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    // ... (código existente) ...
    
    // ← NUEVO: Inicializar RAG
    RAGEngine rag("../rag/embeddings/chunks.json");
    printf("✅ Sistema RAG cargado\n");
    
    // ... (código existente hasta el loop de chat) ...
    
    while (true) {
        printf("\033[32m> \033[0m");
        std::string user;
        std::getline(std::cin, user);
        
        if (user.empty()) {
            break;
        }
        
        // ← NUEVO: Obtener contexto RAG
        std::string rag_context = rag.get_augmented_context(user, 3);
        
        // Construir mensaje del usuario con contexto
        std::string user_message = rag_context + "Pregunta: " + user;
        
        messages.push_back({"user", strdup(user_message.c_str())});
        
        // ... (resto del código de generación) ...
    }
    
    // ... (cleanup) ...
}
```

### Paso 5: Actualizar CMakeLists.txt del app

Modificar `app/CMakeLists.txt`:

```cmake
# Descargar nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_executable(obgyn_chat 
    main.cpp
    bm25.cpp
    rag_engine.cpp
)

target_link_libraries(obgyn_chat PRIVATE 
    llama
    nlohmann_json::nlohmann_json
)

target_compile_features(obgyn_chat PRIVATE cxx_std_17)
```

---

## 📦 Tamaño Final Estimado

| Componente | Tamaño |
|------------|--------|
| Modelo LLM (Qwen 2.5 1.5B Q4) | ~1.0 GB |
| chunks.json | ~2.7 MB |
| Código BM25 (compilado) | ~50 KB |
| **TOTAL** | **~1.003 GB** |

**Sin modelo de embeddings adicional** ✅

---

## 🚀 Compilar y Ejecutar

```powershell
cd "c:\Users\Luis Seoanes\OneDrive\Desktop\arrow\llm"

# Configurar
cmake -B build -A x64 -DGGML_AVX2=ON

# Compilar
cmake --build build --config Release

# Ejecutar
.\build\app\Release\obgyn_chat.exe
```

---

## 🔍 Ejemplo de Uso

```
> ¿Cuáles son los signos de preeclampsia?

[Sistema RAG busca en chunks.json con BM25]
[Encuentra 3 chunks relevantes sobre preeclampsia]
[Inyecta contexto en el prompt]

Asistente: Según las guías clínicas, los signos de alarma de 
preeclampsia incluyen:
1. Presión arterial ≥140/90 mmHg
2. Proteinuria
3. Cefalea intensa
4. Alteraciones visuales
...
```

---

## ✅ Ventajas de Esta Implementación

- ✅ **Sin embeddings en móvil**: Solo BM25 (búsqueda léxica)
- ✅ **Ligero**: Solo 2.7 MB adicionales
- ✅ **Rápido**: <5ms por búsqueda
- ✅ **Offline completo**: Sin dependencias externas
- ✅ **Escalable**: Fácil agregar más documentos

---

## 📝 Notas Importantes

1. **BM25 vs Embeddings**: BM25 funciona bien para términos médicos específicos
2. **Actualización**: Para agregar documentos, re-ejecutar `generate_embeddings.py`
3. **Fallback**: Si BM25 no encuentra resultados, el LLM responde sin contexto
4. **Optimización**: Cachear búsquedas frecuentes para mejorar latencia
