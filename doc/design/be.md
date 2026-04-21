# Decisiones de Stack Backend

## Opciones consideradas

### 1. Python (Kivy / BeeWare)

Python es el lenguaje dominante en desarrollo de inteligencia artificial y ofrece 
un ecosistema maduro para inferencia y procesamiento de lenguaje natural. 
Kivy y BeeWare permiten empaquetar aplicaciones Python para Android.

**Ventajas**
- Ecosistema de IA consolidado (transformers, llama-cpp-python, faster-whisper)
- Curva de aprendizaje baja para equipos con perfil de ciencia de datos
- Prototipado rápido del pipeline de inferencia

**Desventajas**
- El runtime de Python incrementa el tamaño del APK entre 80–120 MB adicionales,
  comprometiendo el límite de 2.5 GB
- Rendimiento en inferencia sobre ARM64 notablemente inferior al código nativo compilado
- Integración con Room, WorkManager y el ciclo de vida de Android requiere 
  bindings no oficiales e inestables
- Publicación en Play Store con runtimes no estándar genera fricciones de revisión

---

### 2. Kotlin Nativo (Android SDK)

Kotlin es el lenguaje oficial de Android y se integra directamente con todo el 
ecosistema del SDK: Room para persistencia, WorkManager para tareas en background, 
ViewModel para gestión de estado y Jetpack Compose para UI.

**Ventajas**
- Integración nativa con Room, WorkManager y el ciclo de vida de Android
- Arquitectura limpia y mantenible mediante capas bien definidas
- Soporte oficial de Google, documentación extensa y comunidad activa
- Compatible con publicación en Play Store sin restricciones

**Desventajas**
- No puede ejecutar directamente modelos cuantizados en formato GGUF
- La inferencia de modelos LLM desde la JVM introduce overhead de memoria 
  y latencia inaceptables para el presupuesto de ≤ 1 GB RAM

---

### 3. C++ Nativo vía JNI (Android NDK)

El Android NDK permite compilar código C++ que se ejecuta directamente sobre el 
hardware del dispositivo. La integración con Kotlin se realiza mediante JNI 
(Java Native Interface). Las librerías llama.cpp y whisper.cpp, desarrolladas en C++, 
son las referencias de facto para inferencia de modelos cuantizados en dispositivos 
con recursos limitados.

**Ventajas**
- Ejecución directa sobre ARM64 con instrucciones NEON y dotprod: mejor rendimiento 
  de inferencia posible en móvil
- Control granular sobre el uso de memoria: el modelo puede cargarse, descargarse 
  y liberarse de forma determinística
- llama.cpp y whisper.cpp son activamente mantenidos y optimizados para modelos 
  GGUF en dispositivos ARM
- Compatible con Play Store y con el ciclo de vida de Android mediante JNI

**Desventajas**
- Mayor complejidad de configuración: requiere toolchain cruzado (NDK + CMake) 
  y gestión manual de memoria
- Depuración más costosa que en Kotlin puro
- La interfaz JNI expone un boundary frágil: errores de nombres de función o tipos 
  producen crashes silenciosos en tiempo de ejecución

---

## Decisión final

Se adopta una **arquitectura híbrida Kotlin + C++ vía JNI**.

Kotlin actúa como lenguaje principal para la lógica de negocio, orquestación del 
pipeline, persistencia local (Room), gestión del historial clínico y tareas 
programadas (WorkManager). C++ vía JNI opera como motor de inferencia exclusivo 
para el modelo de lenguaje y la transcripción de voz offline (whisper.cpp).

Esta decisión responde al cumplimiento simultáneo de cuatro restricciones 
no negociables del proyecto:

| Restricción | Cómo la resuelve la arquitectura híbrida |
|---|---|
| Optimización de RAM en ejecución | C++ permite carga y descarga determinística del modelo; Kotlin no duplica el contexto de inferencia |
| ≤ 2.5 GB tamaño de app | Sin runtime de Python; el .so compilado para ARM64 es < 2 MB sin el modelo |
| 100% offline en Android 10+ | llama.cpp corre localmente sin dependencias de red; Room persiste el historial sin sincronización |
| Publicación en Google Play | NDK y Kotlin son stacks oficiales de Android; sin runtimes no estándar |

Ninguna de las tres opciones evaluadas de forma aislada satisface el conjunto 
completo de restricciones. La combinación Kotlin + C++ es la única alternativa 
que lo hace.