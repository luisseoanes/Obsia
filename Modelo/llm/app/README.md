# Chatbot obstetricia / salud materna (MVP)

Aplicación de consola que usa el modelo GGUF en local para responder consultas de apoyo clínico en obstetricia y salud materna.

## Estructura del proyecto

- **app/** – Código fuente del chatbot (main.cpp, RAG, BM25).
- **llama.cpp/** – Dependencia necesaria: se compila como biblioteca (no se construyen herramientas ni ejemplos). No borrar; el `CMakeLists.txt` la usa con `LLAMA_STANDALONE OFF`.
- **rag/** – Base de conocimiento, embeddings y scripts Python para generarlos.
- **models/** – Modelos GGUF (ej. Qwen).
- **build/** – Generado por CMake; contiene el .exe. Puede borrarse y recrearse con `cmake -B build`.

Archivos/carpetas que **no** forman parte del código y se ignoran (ver `.gitignore`): `build/`, `CMakeFiles/`, `llama-build/`, `CMakeCache.txt`, `*.vcxproj` generados, etc. Para limpiar todo: ejecutar `.\clean.ps1` en la raíz.

## Compilar

Abre **Developer PowerShell for VS 2022** (o Developer Command Prompt) y ejecuta:

```powershell
cd "c:\Users\Luis Seoanes\OneDrive\Desktop\arrow\llm"

# Configurar (desde la raíz del proyecto llm)
cmake -B build -A x64 -DGGML_AVX2=ON

# Compilar
cmake --build build --config Release
```

El ejecutable queda en: `build\app\Release\obgyn_chat.exe`

## Ejecutar

Desde la carpeta **llm** (raíz del proyecto), para que encuentre el modelo por defecto:

```powershell
cd "c:\Users\Luis Seoanes\OneDrive\Desktop\arrow\llm"
.\build\app\Release\obgyn_chat.exe
```

O indicando la ruta del modelo:

```powershell
.\build\app\Release\obgyn_chat.exe -m models\qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Opciones principales:

- `-m ruta`   Modelo GGUF
- `-c N`     Tamaño de contexto (por defecto: 4096)
- `-ngl N`   Capas en GPU; 0 = solo CPU (por defecto: 0)
- `-k N`     Fragmentos RAG a inyectar (1–10; por defecto: 3; menos = más rápido)
- `-t N`     Hilos CPU (por defecto: 2)
- `-b N`     Batch size para prefill (512 por defecto; 1024 suele acelerar sin subir CPU)
- `-ub N`    Micro-batch (por defecto: 512)

Escribe tu pregunta y Enter. Línea vacía para salir.

---

## Acelerar consultas sin subir uso de CPU

Para que las respuestas lleguen antes **sin aumentar** el uso de CPU:

1. **Menos fragmentos RAG**  
   `-k 3` (o `-k 2`): menos texto en el prompt → menos tokens que procesar en cada consulta.

2. **Batch más grande en prefill**  
   `-b 1024` o `-b 2048`: se procesan más tokens por llamada a `llama_decode` durante el prefill → menos pasadas y menos sobrecarga, mismo uso de CPU.

3. **Modelo más pequeño**  
   Usar el 0.5B en lugar del 1.5B: menos parámetros → menos trabajo por token (a cambio de algo de calidad).

4. **Contexto más corto**  
   `-c 2048`: menos memoria KV y menos trabajo si no necesitas conversaciones muy largas.

5. **Combinación recomendada para máxima velocidad (mismo CPU)**  
   ```powershell
   .\build\app\Release\obgyn_chat.exe -m models\qwen2.5-0.5b-instruct-q4_0.gguf -k 3 -b 1024 -c 2048
   ```

La app ya usa **Flash Attention** y **caché KV en 8 bits** (Q8_0), que reducen uso de RAM y ayudan a la velocidad sin subir CPU.
