# ObsIA: IA Conversacional Offline para Apoyo Clínico

**ObsIA** es un MVP de inteligencia artificial conversacional diseñado para operar totalmente **offline** en dispositivos móviles. Su objetivo principal es brindar apoyo a equipos clínicos en obstetricia y salud materna, especialmente en contextos de atención inmediata donde la conectividad es limitada.

## Propósito del Proyecto

El sistema funciona como una herramienta de apoyo a la decisión clínica, utilizando modelos de lenguaje locales y una base de conocimiento validada para ofrecer respuestas estructuradas y trazables.

---

## Características Técnicas

- **Chat Clínico Offline:** Procesamiento de texto sin necesidad de internet para garantizar disponibilidad en terreno.
- **Motor de Reglas Clínicas:** Sistema determinístico para el manejo de escenarios críticos y urgencias.
- **RAG (Retrieval-Augmented Generation):** Base de conocimientos clínica preindexada y embebida para reducir alucinaciones.
- **Inferencia Local Optimizada:** Ejecución de modelos LLM mediante optimización para arquitectura ARM.

---

## Limitaciones del Proyecto

Dado que **ObsIA** se ejecuta en un entorno móvil offline y maneja información sensible, existen restricciones importantes:

### 1. Alcance Clínico

- **No es un sistema de diagnóstico:** El software se define exclusivamente como una herramienta de apoyo a la decisión clínica, no como un sustituto del diagnóstico profesional autónomo.
- **Base de Conocimiento:** Las respuestas están limitadas a la base de conocimiento clínica cargada y versionada en el sistema.

### 2. Restricciones Técnicas (Hardware)

- **Capacidad del Modelo:** Debido al uso en dispositivos móviles, el modelo está limitado a aproximadamente 1B de parámetros para asegurar la viabilidad técnica.
- **Recursos del Dispositivo:** El rendimiento está sujeto a limitaciones severas de memoria RAM, capacidad de procesamiento (CPU) y consumo de batería del celular.
- **Optimización Extrema:** La necesidad de ejecución offline requiere una cuantización y optimización agresiva del modelo y del sistema RAG.

### 3. Implementación y Entorno

- **Diversidad de Dispositivos:** El correcto funcionamiento puede variar según el hardware del fabricante, requiriendo un esfuerzo mayor de pruebas multidispositivo.
- **Actualizaciones:** Al ser una aplicación estrictamente offline, las actualizaciones de la base de conocimiento requieren una reinstalación o actualización manual del paquete de datos.

---

## Stack Tecnológico

Para cumplir con las restricciones severas de hardware en móviles, el proyecto utiliza:

- **Lenguajes:** Kotlin (Android nativo) y C/C++ para el motor de IA
- **Interoperabilidad:** Integración mediante **JNI** (Java Native Interface).
- **IA:** Modelos de lenguaje cuantizados (Instruct y Lama.cpp).
- **Empaquetado:** Formato APK/OBB para instalación completa fuera de línea.

---

## Equipo de Desarrollo

| **Miembro** | **Rol** | **Responsabilidad Principal** |
| --- | --- | --- |
| **Julián** | Frontend Developer | Interfaz de usuario (UI) y experiencia de chat en Android. |
| **Luis** | AI Engineer/lead project | Inferencia nativa en C++, optimización del modelo y RAG. |
| **Valentina** | Backend Developer | Orquestación, sistema RAG y lógica de negocio clínica. |
| **Rafa** | Backend Developer/QA | Integración JNI, motor de reglas y persistencia local. |

---

## 📂 Tabla de Navegación

| 🚀 Sección                                                         | 📄 Descripción                                                                                                 |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| 📚 [Documentación principal](./doc/index.md)                      | Carpeta principal donde se encuentra toda la documentación del proyecto.                                       |
| 📊 [Carpeta de análisis](./doc/analysis/index.md)                          | Carpeta que contiene toda la documentación relacionada con la fase de análisis del sistema.                    |
| 🧭 [Navegación de análisis](./doc/analysis/index.md)               | Documento de navegación hacia los requisitos funcionales y no funcionales del sistema.                         |
| 📋 [Requisitos Funcionales](./doc/analysis/requirements-fn.md)     | Documento que describe las funcionalidades que el sistema debe proporcionar.                                   |
| ⚙️ [Requisitos No Funcionales](./doc/analysis/requirements-nfn.md) | Documento que describe las características de calidad del sistema como rendimiento, seguridad y escalabilidad. |


