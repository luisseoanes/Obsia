# Sistema RAG para Asistente Médico Móvil

Sistema de Retrieval-Augmented Generation (RAG) optimizado para dispositivos móviles con base de datos vectorial pre-computada.

## 📁 Estructura

```
rag/
├── knowledge_base/          # PDFs de guías clínicas (input)
├── embeddings/              # Base de datos vectorial (output)
│   ├── chunks.json         # Fragmentos de texto indexados
│   ├── index.faiss         # Índice vectorial FAISS cuantizado
│   ├── embeddings.npy      # Embeddings raw (opcional)
│   └── metadata.json       # Metadata del sistema
└── scripts/
    ├── generate_embeddings.py  # Script principal
    └── requirements.txt        # Dependencias Python
```

## 🚀 Generación de Embeddings (en PC)

### 1. Instalar dependencias

```powershell
cd "c:\Users\Luis Seoanes\OneDrive\Desktop\arrow\llm\rag\scripts"
pip install -r requirements.txt
```

### 2. Ejecutar generación

```powershell
python generate_embeddings.py
```

El script procesará automáticamente todos los PDFs en `knowledge_base/` y generará:
- ✅ Chunks inteligentes respetando párrafos médicos
- ✅ Embeddings con modelo multilingüe optimizado
- ✅ Índice FAISS cuantizado (IVF-PQ) para móvil
- ✅ Pruebas de búsqueda semántica

## 📊 Características del Sistema

### Modelo de Embeddings
- **Modelo**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensión**: 384
- **Idiomas**: Español, inglés, y 50+ idiomas
- **Tamaño**: ~120 MB (solo para generación en PC)

### Chunking Inteligente
- **Tamaño**: 400 caracteres por chunk
- **Overlap**: 50 caracteres entre chunks
- **Estrategia**: Respeta párrafos médicos completos

### Índice FAISS
- **Tipo**: IVF-PQ (Inverted File + Product Quantization)
- **Compresión**: ~90% reducción vs. embeddings raw
- **Búsqueda**: k-NN con similitud coseno
- **Optimización**: Cuantización para móvil

## 🔍 Ejemplo de Uso

```python
import faiss
import json
import numpy as np

# Cargar índice y chunks
index = faiss.read_index("embeddings/index.faiss")
with open("embeddings/chunks.json") as f:
    chunks = json.load(f)

# Buscar (requiere generar embedding del query)
query_embedding = model.encode(["¿Signos de preeclampsia?"])
distances, indices = index.search(query_embedding, k=3)

# Recuperar chunks relevantes
for idx in indices[0]:
    print(chunks[idx]["text"])
```

## 📦 Tamaño Estimado

| Componente | Tamaño Aprox. |
|------------|---------------|
| chunks.json | 5-20 MB |
| index.faiss | 10-50 MB |
| embeddings.npy | 20-100 MB |
| **Total para móvil** | **15-70 MB** |

> **Nota**: Solo `chunks.json` e `index.faiss` son necesarios en móvil.

## 🔧 Integración en C++

Ver documentación en `/app` para integrar FAISS en el código C++.

## 📝 Notas Importantes

1. **Generación solo en PC**: Los embeddings se generan una sola vez en PC
2. **Sin modelo en móvil**: El móvil solo hace búsqueda k-NN, no genera embeddings
3. **Actualización**: Para agregar documentos, re-ejecutar `generate_embeddings.py`
4. **Cuantización**: El índice IVF-PQ reduce precisión ~2-5% pero ahorra 90% de espacio
