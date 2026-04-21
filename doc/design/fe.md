# Análisis y Decisión de Framework Frontend

## Introducción

El desarrollo del frontend del proyecto se enfoca en una aplicación móvil Android. 
La elección de la tecnología impacta directamente en el rendimiento, mantenibilidad, escalabilidad y experiencia del usuario.

Se analizaron cuatro enfoques principales:

1. Desarrollo Nativo Android (XML + View System)
2. Jetpack Compose
3. Flutter
4. React Native

## Opciones Consideradas

### 1️⃣ Android Nativo (XML + View System)

- Lenguaje: Kotlin
- Entorno: Android Studio
- Enfoque tradicional basado en layouts XML.

#### Ventajas
- Alto rendimiento (100% nativo).
- Control total sobre componentes.
- Amplia documentación oficial.
- Estabilidad probada en proyectos reales.

#### Desventajas
- Código más extenso.
- Separación entre XML y lógica puede aumentar complejidad.
- Enfoque menos moderno comparado con Compose.

### 2️⃣ Jetpack Compose

- Lenguaje: Kotlin
- UI declarativa moderna para Android.

#### Ventajas
- Código más limpio y conciso.
- Arquitectura moderna y reactiva.
- Mayor productividad.
- Recomendado oficialmente por Google.
- Mejor mantenibilidad a largo plazo.

#### Desventajas
- Tecnología relativamente reciente.
- Requiere adaptación al paradigma declarativo.

### 3️⃣ Flutter

- Lenguaje: Dart
- Framework multiplataforma (Android e iOS).

#### Ventajas
- Una sola base de código.
- Desarrollo rápido.
- Buen rendimiento.

#### Desventajas
- No utiliza Kotlin.
- No es desarrollo nativo puro.
- Introduce dependencia a un framework externo.

### 4️⃣ React Native

- Lenguaje: JavaScript
- Desarrollo multiplataforma.

#### Ventajas
- Comunidad amplia.
- Reutilización de código entre plataformas.

#### Desventajas
- Rendimiento inferior al desarrollo nativo.
- Dependencia de puentes nativos.
- No alineado con Kotlin ni Android Studio.

## Comparativa

| Criterio | XML Nativo | Jetpack Compose | Flutter | React Native |
|----------|------------|----------------|----------|--------------|
| Rendimiento | Muy alto | Muy alto | Alto | Medio |
| Modernidad | Media | Alta | Alta | Alta |
| Alineación con Kotlin | Total | Total | Ninguna | Ninguna |
| Escalabilidad | Alta | Muy alta | Alta | Media |
| Mantenibilidad | Media | Alta | Media | Media |

## 📌 Decisión Final

Se decidió utilizar **Kotlin + Android Studio + Jetpack Compose** para el desarrollo frontend.

### Justificación

1. Desarrollo 100% nativo Android.
2. Integración directa con el ecosistema oficial de Google.
3. Arquitectura moderna declarativa.
4. Mayor claridad y mantenibilidad del código.
5. Coherencia tecnológica con el stack definido para el proyecto.
6. Mejor proyección académica y profesional en el entorno Android.

## Conclusión

Jetpack Compose junto con Kotlin representa la solución más adecuada para el desarrollo frontend del proyecto, ofreciendo rendimiento nativo, modernidad arquitectónica y coherencia tecnológica.

## Referencias

Google Developers. (s.f.). Android architecture recommendations. https://developer.android.com/topic/architecture
Android Developers. (s.f.). UI toolkit documentation. https://developer.android.com/guide/topics/ui
