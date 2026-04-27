# ObsIA: Offline Conversational AI for Clinical Support

**ObsIA** is a Conversational AI MVP (Minimum Viable Product) designed to operate completely **offline** on mobile devices. Its primary goal is to provide decision support for clinical teams in obstetrics and maternal health, particularly in immediate care settings where connectivity is limited or non-existent.

---

## Project Purpose

The system serves as a clinical decision support tool, utilizing local Large Language Models (LLMs) and a validated clinical knowledge base to provide structured, traceable, and reliable responses at the point of care.

---

## Technical Features

- **Offline Clinical Chat:** Full text processing without internet requirements to ensure 100% availability in the field.
- **Clinical Rule Engine:** A deterministic system for handling critical scenarios and emergencies with high reliability.
- **RAG (Retrieval-Augmented Generation):** A pre-indexed, embedded clinical knowledge base that reduces model hallucinations by anchoring responses in medical literature.
- **Optimized Local Inference:** High-performance execution of LLMs through specialized optimizations for ARM (mobile) architectures.

---

## Architecture and Project Structure

While the standard Android project structure usually keeps everything within the `app` folder, **ObsIA** separates the high-performance AI logic to maintain a clean distinction between the mobile interface and the inference engine.

### 📂 Directory Overview

| Directory | Description |
| --- | --- |
| `app/` | **Android Application Layer**: Built with Kotlin. Handles the UI (Jetpack Compose), local persistence, and the orchestration of clinical rules. |
| `Modelo/` | **AI Core Ecosystem**: This directory centralizes the intelligence of the system. |
| `Modelo/llm/` | **Development & RAG Lab**: Contains the scripts for generating embeddings, pre-indexing clinical documents, and testing the retrieval logic. |
| `Modelo/modeloFinal/` | **Production Inference Engine**: The high-performance C++ implementation based on `llama.cpp`. This is where the cross-compilation for mobile (ARM64) occurs. |

### Why is the C++ model in `Modelo`?

The project follows a "Separation of Concerns" principle:
1.  **Development vs. Production**: The `Modelo/llm` folder is where the model is tested, quantized, and where the RAG system is developed using Python and other tools.
2.  **Library Generation (JNI Bridge)**: The `Modelo/modeloFinal` folder contains the specific C++ source code that is compiled into a native library (.so). This library is then consumed by the Android app via **JNI (Java Native Interface)**. 
3.  **Portability**: By keeping the AI logic in a separate C++ module, we ensure that the core engine can be optimized, updated, or even ported to other platforms without affecting the mobile application's UI/UX logic.

---

## Project Limitations

Since **ObsIA** runs in a strictly offline mobile environment and handles sensitive clinical information, it has the following constraints:

### 1. Clinical Scope
- **Not a Diagnostic System**: The software is strictly a clinical decision support tool and is not a substitute for professional medical diagnosis.
- **Knowledge Base Boundaries**: Responses are strictly limited to the clinical knowledge base loaded and versioned within the system.

### 2. Technical Constraints (Hardware)
- **Model Size**: Limited to models with approximately 1B to 3B parameters to ensure technical viability on standard mobile hardware.
- **Resource Management**: Performance is highly dependent on device RAM, CPU capacity, and thermal throttling.
- **Aggressive Optimization**: Requires 4-bit (or lower) quantization and extreme RAG optimization to maintain responsiveness.

### 3. Deployment & Updates
- **Device Diversity**: Performance may vary significantly between different hardware manufacturers.
- **Manual Updates**: Knowledge base updates require a manual package update or app reinstallation due to the offline nature of the system.

---

## Tech Stack

- **Languages**: Kotlin (Native Android) and C/C++ (Inference Engine).
- **Interoperability**: **JNI** (Java Native Interface) for high-speed communication between Kotlin and C++.
- **AI Core**: `llama.cpp` for GGUF model inference.
- **Data Format**: Quantized models and pre-computed Faiss/Binary indexes for RAG.
- **Build System**: Gradle (Android) and CMake (C++).

---

## Development Team

| Member | Role | Primary Responsibility |
| --- | --- | --- |
| **Julián** | Frontend Developer | UI/UX and Chat experience in Android. |
| **Luis** | AI Engineer / Lead | Native C++ inference, model optimization, and RAG architecture. |
| **Valentina** | Backend Developer | Orchestration, RAG system logic, and clinical business logic. |
| **Rafa** | Backend/QA | JNI Integration, Rule Engine, and local persistence. |

---

## 📂 Navigation Table

| 🚀 Section | 📄 Description |
| --- | --- |
| 📚 [Main Documentation](./doc/index.md) | Main folder containing all project documentation. |
| 📊 [Analysis Folder](./doc/analysis/index.md) | Documentation related to the system analysis phase. |
| 🧭 [Analysis Navigation](./doc/analysis/index.md) | Navigation guide for Functional and Non-Functional requirements. |
| 📋 [Functional Requirements](./doc/analysis/requirements-fn.md) | Description of the system's core features. |
| ⚙️ [Non-Functional Requirements](./doc/analysis/requirements-nfn.md) | Quality attributes such as performance, security, and scalability. |
