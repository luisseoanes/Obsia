# ObsIA: The First Fully Offline AI for Maternal & Obstetric Care

![Status](https://img.shields.io/badge/Status-MVP-orange)
![Platform](https://img.shields.io/badge/Platform-Android-green)
![Intelligence](https://img.shields.io/badge/Engine-C%2B%2B%20%2F%20llama.cpp-blue)
![Connectivity](https://img.shields.io/badge/Internet-Not%20Required-red)

**ObsIA** is a groundbreaking clinical decision support system designed to provide high-level AI capabilities in the most challenging environments. Built for obstetricians and maternal health teams, ObsIA operates **100% offline**, ensuring that critical medical knowledge is available even in remote areas without cellular or satellite coverage.

---

## 🌟 The Vision

In rural and emergency clinical settings, every second counts. Lack of connectivity often prevents access to modern AI assistants. **ObsIA** bridges this gap by bringing the power of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) directly into the clinician's pocket.

### Core Objectives:
1.  **Empower Rural Healthcare**: Provide expert-level clinical support where specialists are unavailable.
2.  **Absolute Privacy**: No data leaves the device, complying with the strictest medical data regulations.
3.  **Instant Availability**: Zero latency from network round-trips; the AI is as fast as the local processor.

---

## 🏗️ System Architecture

ObsIA uses a **Hybrid Dual-Core Architecture** to balance modern Android UI flexibility with raw C++ performance.

```mermaid
graph TD
    subgraph "Android Application (Kotlin/Compose)"
        UI[User Interface] --> Orchestrator[Query Orchestrator]
        Orchestrator --> RuleEngine[Clinical Rule Engine]
    end

    subgraph "Native Inference Engine (C++/JNI)"
        JNI[JNI Bridge] --> LlamaCpp[llama.cpp Core]
        LlamaCpp --> RAG[RAG Retrieval System]
        RAG --> KnowledgeBase[(Local Clinical Docs)]
        LlamaCpp --> LLM[Quantized LLM]
    end

    Orchestrator <==> JNI
```

### 1. The Kotlin/Compose Frontend (`app/`)
Handles the user experience, chat state management, and the **Clinical Rule Engine**. This engine acts as a deterministic safety layer, ensuring that critical emergency protocols (like Eclampsia or Postpartum Hemorrhage) are handled with 100% predictable logic before the AI even starts "thinking."

### 2. The Native C++ Backend (`Modelo/`)
This is the heart of the system. We use a customized fork of `llama.cpp` optimized for ARM64 (Android) architectures.
- **Model Development (`Modelo/llm/`)**: This is our research lab. We use Python and specialized tools here to prepare documents, generate Faiss/Binary embeddings, and quantize the models (from FP16 to 4-bit) to fit mobile RAM constraints.
- **Production Engine (`Modelo/modeloFinal/`)**: This folder contains the specialized C++ code that compiles into a `.so` (shared object) library. It is designed for maximum speed and efficient memory management.

---

## 🧠 Intelligent Pillars

### 📋 Deterministic Rule Engine
Before the LLM processes a query, a rule-based system scans for "red flag" clinical symptoms. If a life-threatening emergency is detected (e.g., severe hypertension in pregnancy), the system immediately triggers a validated clinical protocol response.

### 📚 RAG (Retrieval-Augmented Generation)
To eliminate "AI hallucinations," ObsIA doesn't rely solely on the model's training data. Instead:
1.  **Indexing**: Clinical guides (PDFs) are chunked and converted into vector embeddings during development.
2.  **Retrieval**: When a clinician asks a question, the system finds the most relevant text snippets from the local knowledge base.
3.  **Generation**: The LLM uses these snippets as context to generate a response that is grounded in clinical evidence.

---

## 📂 Project Structure

| Path | Purpose |
| --- | --- |
| `app/` | Source code for the Android application (Kotlin). |
| `Modelo/` | Central hub for all AI-related logic. |
| `Modelo/llm/` | Research scripts, RAG indexing tools, and model quantization tests. |
| `Modelo/modeloFinal/` | Native C++ inference engine source and JNI interface. |
| `doc/` | Comprehensive technical documentation, analysis, and design diagrams. |

---

## 🛠️ Setup & Development

### Prerequisites:
- **Android Studio Jellyfish** (or newer).
- **Android NDK** (Side-by-side) & **CMake**.
- A device with at least **8GB RAM** for optimal inference (6GB minimum).

### Build Instructions:
1.  **Clone the Repo**: Ensure submodules are initialized.
2.  **Native Build**: Navigate to `Modelo/modeloFinal` and run the provided build scripts or let Android Studio handle the CMake sync.
3.  **Model Loading**: Place your quantized `.gguf` model in the `app/src/main/assets/` directory (ensure it is listed in `.gitignore` to avoid large file errors).
4.  **Run**: Build the APK and deploy to your device.

---

## 👥 The Team

| Name | Role | Focus |
| --- | --- | --- |
| **Luis** | **AI Lead & Engineer** | Native C++ optimization, JNI Bridge, and Model Architecture. |
| **Julián** | **Lead Frontend** | UI/UX Design, Jetpack Compose, and Chat Lifecycle. |
| **Valentina** | **Business Logic/RAG** | Clinical knowledge base indexing and RAG orchestration. |
| **Rafa** | **Backend & QA** | Rule Engine development, persistence, and JNI integration testing. |

---

## ⚖️ Legal Disclaimer

**ObsIA is a Research MVP.** It is intended for educational and clinical decision support purposes only. It is **NOT** a replacement for professional medical judgment. All generated responses should be verified against standard clinical protocols by a qualified healthcare professional.

---
*Built with ❤️ for the health of mothers and babies.*
