#!/usr/bin/env python3
"""
Script de prueba para validar la búsqueda semántica en la base de datos RAG
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def test_rag_search():
    """Prueba el sistema RAG con queries de ejemplo"""
    
    # Rutas
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDINGS_DIR = BASE_DIR / "embeddings"
    
    print("="*70)
    print("🔍 PRUEBA DE BÚSQUEDA SEMÁNTICA - SISTEMA RAG")
    print("="*70)
    
    # Cargar modelo
    print("\n🧠 Cargando modelo de embeddings...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ Modelo cargado")
    
    # Cargar índice FAISS
    print("\n📊 Cargando índice FAISS...")
    index_path = EMBEDDINGS_DIR / "index.faiss"
    index = faiss.read_index(str(index_path))
    print(f"✅ Índice cargado: {index.ntotal} vectores, dimensión {index.d}")
    
    # Cargar chunks
    print("\n📄 Cargando chunks...")
    chunks_path = EMBEDDINGS_DIR / "chunks.json"
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✅ {len(chunks)} chunks cargados")
    
    # Queries de prueba
    test_queries = [
        "¿Cuáles son los signos de alarma en preeclampsia?",
        "Manejo del trabajo de parto prematuro",
        "Contraindicaciones para parto vaginal",
        "Cuidados prenatales en el primer trimestre",
        "Complicaciones del embarazo gemelar"
    ]
    
    print("\n" + "="*70)
    print("🔎 RESULTADOS DE BÚSQUEDA")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print(f"{'─'*70}")
        
        # Generar embedding del query
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # Buscar top-3 chunks más similares
        k = 3
        distances, indices = index.search(query_embedding, k)
        
        # Mostrar resultados
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            chunk = chunks[idx]
            text = chunk["text"]
            source = chunk["metadata"]["source"]
            chunk_id = chunk["metadata"]["chunk_id"]
            
            # Truncar texto para visualización
            display_text = text[:250] + "..." if len(text) > 250 else text
            
            print(f"\n  [{rank}] Score: {score:.4f} | Source: {source} (chunk {chunk_id})")
            print(f"      {display_text}")
    
    print("\n" + "="*70)
    print("✅ Prueba completada exitosamente")
    print("="*70)
    
    # Estadísticas
    print("\n📊 ESTADÍSTICAS DEL SISTEMA")
    print(f"  • Total de chunks:        {len(chunks)}")
    print(f"  • Dimensión embeddings:   {index.d}")
    print(f"  • Tipo de índice:         {type(index).__name__}")
    print(f"  • Tamaño chunks.json:     {chunks_path.stat().st_size / 1024:.2f} KB")
    print(f"  • Tamaño index.faiss:     {index_path.stat().st_size / 1024:.2f} KB")
    print(f"  • Total:                  {(chunks_path.stat().st_size + index_path.stat().st_size) / 1024:.2f} KB")


if __name__ == "__main__":
    test_rag_search()
