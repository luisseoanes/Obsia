## RF-01 – Ingreso de consulta obstétrica por texto

**Descripción**
La aplicación debe permitir a la usuaria ingresar una consulta obstétrica mediante un campo de texto libre.

**Criterios de aceptación**

```gherkin
Feature: Ingreso de consulta por texto

Scenario: Envío de consulta escrita
  DADO que la aplicación está abierta
  CUANDO la usuaria escribe una consulta válida
  Y presiona el botón enviar
  ENTONCES el sistema recibe la consulta para su procesamiento

Scenario: Intento de envío sin texto
  DADO que el campo de texto está vacío
  CUANDO la usuaria presiona enviar
  ENTONCES el sistema bloquea el envío
  Y muestra el mensaje "Debe ingresar una consulta"

Scenario: Consulta supera el límite permitido
  DADO que la usuaria escribe más de 500 caracteres
  CUANDO intenta enviar la consulta
  ENTONCES el sistema impide el envío
  Y muestra un mensaje indicando que se superó el límite permitido
```

**Prioridad:** P0
**Puntos :** 3

---

## RF-02 – Clasificación temática de la consulta

**Descripción**
El sistema debe clasificar si la consulta pertenece al dominio obstétrico.

**Criterios de aceptación**

```gherkin
Feature: Clasificación de dominio

Scenario: Consulta enviada
  DADO que la usuaria envía una consulta
  CUANDO el sistema la analiza
  ENTONCES determina si pertenece al dominio obstétrico
```

**Prioridad:** P0
**Puntos :** 3

---

## RF-03 – Rechazo de consulta fuera del dominio

**Descripción**
Si la consulta no pertenece al dominio obstétrico, el sistema debe rechazarla sin generar contenido clínico.

**Criterios de aceptación**

```gherkin
Feature: Rechazo fuera de alcance

Scenario: Consulta no obstétrica
  DADO que la consulta fue clasificada como fuera de dominio
  CUANDO el sistema responde
  ENTONCES muestra un mensaje indicando que está fuera del alcance de la aplicación
  Y no genera respuesta clínica
```

**Prioridad:** P0
**Puntos :** 2

---

## RF-04 – Búsqueda en base de casos clínicos predefinidos

**Descripción**
El sistema debe verificar si la consulta coincide con un caso clínico predefinido almacenado localmente.

**Criterios de aceptación**

```gherkin
Feature: Búsqueda en base predefinida

Scenario: Coincidencia encontrada
  DADO que la consulta es obstétrica
  CUANDO existe coincidencia en la base de casos clínicos
  ENTONCES el sistema selecciona la respuesta almacenada
```

**Prioridad:** P0
**Puntos :** 3

---

## RF-05 – Generación de respuesta mediante modelo local

**Descripción**
Si no existe coincidencia en la base de casos, el sistema debe generar la respuesta utilizando el modelo de lenguaje local.

**Criterios de aceptación**

```gherkin
Feature: Ejecución de modelo local

Scenario: Sin coincidencia en base predefinida
  DADO que la consulta es obstétrica
  Y no existe coincidencia en la base de casos
  CUANDO el sistema procesa la consulta
  ENTONCES ejecuta el modelo de lenguaje local
  Y genera una respuesta clínica
```

**Prioridad:** P0
**Puntos :** 5

---

## RF-06 – Detección de emergencia obstétrica

**Descripción**
El sistema debe identificar si la consulta contiene indicadores de emergencia obstétrica.

**Criterios de aceptación**

```gherkin
Feature: Identificación de emergencia

Scenario: Consulta con indicadores de riesgo
  DADO que la consulta contiene términos asociados a riesgo vital
  CUANDO el sistema la analiza
  ENTONCES la clasifica como emergencia
```

**Prioridad:** P0
**Puntos :** 3

---

## RF-07 – Etiquetado del nivel de urgencia

**Descripción**
El sistema debe incluir una etiqueta visible indicando el nivel de urgencia de la respuesta.

**Criterios de aceptación**

```gherkin
Feature: Etiquetado de urgencia

Scenario: Respuesta generada
  DADO que existe una clasificación de urgencia
  CUANDO el sistema muestra la respuesta
  ENTONCES incluye una etiqueta visible "URGENTE" o "RUTINA"
```

**Prioridad:** P0
**Puntos :** 2

---

## RF-08 – Inclusión de recomendaciones inmediatas en emergencias

**Descripción**
Cuando la consulta sea clasificada como emergencia, la respuesta debe incluir acciones inmediatas.

**Criterios de aceptación**

```gherkin
Feature: Respuesta prioritaria ante emergencia

Scenario: Emergencia confirmada
  DADO que la consulta fue clasificada como emergencia
  CUANDO el sistema genera la respuesta
  ENTONCES incluye recomendaciones de acción inmediata
```

**Prioridad:** P0
**Puntos :** 3

---

## RF-09 – Visualización del historial de conversaciones

**Descripción**
El sistema debe registrar automáticamente cada par consulta-respuesta generado y permitir al profesional de salud consultarlo posteriormente en una pantalla de historial de solo lectura. El historial no es interactivo ni reutilizable.

**Criterios de aceptación**

```gherkin
Feature: Registro y visualización del historial de conversaciones

Scenario: Almacenamiento tras respuesta generada
  DADO que el sistema ha generado una respuesta clínica
  CUANDO la respuesta es presentada al usuario
  ENTONCES el sistema almacena el par consulta-respuesta en el historial local
  Y registra la marca temporal del momento de generación
  Y registra el nivel de urgencia clasificado
  Y no solicita ninguna acción al usuario para completar el registro

Scenario: Visualización y acceso al historial existente
  DADO que existen registros previos en el historial
  CUANDO el usuario accede a la pantalla de historial
  ENTONCES el sistema muestra las conversaciones en orden cronológico 
  Y cada entrada muestra la consulta, la respuesta y el nivel de urgencia
  Y ninguna entrada permite ser seleccionada para reenvío o edición

Scenario: Independencia entre sesiones
  DADO que existen registros previos en el historial
  CUANDO el usuario envía una nueva consulta
  ENTONCES el sistema procesa la consulta sin utilizar información de registros anteriores
  Y la respuesta generada es independiente del contenido del historial
```

**Prioridad:** P0
**Puntos:** 5

---

## RF-10 – Construcción de pantalla Home (Aterrizaje)

**Descripción**
La aplicación debe contar con una pantalla principal (Home) que actúe como el punto de aterrizaje inmediato al abrir la aplicación, permitiendo al usuario iniciar una consulta de forma directa.

**Criterios de aceptación**

```gherkin
Feature: Pantalla de inicio de la aplicación

  Scenario: Aterrizaje al abrir la app
    DADO que la usuaria abre la aplicación
    ENTONCES el sistema muestra inmediatamente la pantalla de Home
    Y no se presentan pantallas de carga o intermedias innecesarias
    Y la pantalla ofrece las opciones para iniciar una consulta nueva
```

**Prioridad:** P0
**Puntos:** 2

---

## RF-11 – Captura de consulta mediante voz

**Descripción**
La aplicación debe permitir capturar una consulta mediante grabación de audio utilizando el micrófono del dispositivo.

**Criterios de aceptación**

```gherkin
Feature: Captura de audio

Scenario: Inicio de grabación
  DADO que el micrófono está habilitado
  CUANDO la usuaria presiona el botón de grabación
  ENTONCES el sistema inicia la captura de audio
```

**Prioridad:** P1
**Puntos :** 3

---

## RF-12 – Transcripción local de audio

**Descripción**
El sistema debe transcribir el audio capturado a texto utilizando procesamiento local sin conexión a internet.

**Criterios de aceptación**

```gherkin
Feature: Transcripción offline

Scenario: Conversión de audio a texto
  DADO que existe una grabación finalizada
  CUANDO el sistema procesa el audio
  ENTONCES genera una transcripción en texto
  Y no realiza comunicaciones externas
```

**Prioridad:** P1
**Puntos :** 5

---

## RF-13 – Solicitud de permiso de micrófono

**Descripción**
Antes de iniciar la captura de audio, el sistema debe verificar si el permiso
de micrófono ha sido concedido y solicitarlo al usuario en caso de no estarlo.

**Criterios de aceptación**
```gherkin
Feature: Gestión del permiso de micrófono

Scenario: Permiso de micrófono no concedido
  DADO que el permiso de micrófono no ha sido otorgado
  CUANDO la usuaria presiona el botón de grabación
  ENTONCES el sistema solicita el permiso de micrófono al usuario

Scenario: Permiso de micrófono denegado
  DADO que la usuaria deniega el permiso de micrófono
  CUANDO el sistema recibe la respuesta
  ENTONCES deshabilita la funcionalidad de captura por voz
  Y mantiene disponible la entrada por texto
```

**Prioridad:** P1
**Puntos:** 2

---

## RF-14 – Control de tiempo máximo de inferencia

**Descripción**
El sistema debe cancelar la generación de respuesta si el modelo excede el tiempo máximo definido.

**Criterios de aceptación**

```gherkin
Feature: Control de tiempo de procesamiento

Scenario: Modelo excede tiempo permitido
  DADO que el modelo está procesando una consulta
  CUANDO supera el tiempo máximo configurado
  ENTONCES el sistema cancela la ejecución
  Y muestra un mensaje de error controlado
```

**Prioridad:** P1
**Puntos :** 3

---

## RF-15 – Formato directo de la respuesta clínica

**Descripción**
El sistema debe generar respuestas con formato clínico directo, iniciando
inmediatamente con el contenido relevante para el profesional de salud.

**Criterios de aceptación**
```gherkin
Feature: Formato directo de respuesta

Scenario: Generación de respuesta clínica
  DADO que el sistema ha procesado una consulta obstétrica
  CUANDO genera la respuesta
  ENTONCES la respuesta inicia directamente con contenido clínico relevante
  Y no incluye saludos ni expresiones conversacionales
```

**Prioridad:** P1
**Puntos:** 2

---

## RF-16 – Respuesta en un único turno

**Descripción**
Cada consulta debe generar una única respuesta completa sin solicitar información adicional.

**Criterios de aceptación**

```gherkin
Feature: Respuesta única

Scenario: Respuesta generada
  DADO que la usuaria envía una consulta
  CUANDO el sistema responde
  ENTONCES entrega una única respuesta completa
  Y no solicita datos adicionales
```

**Prioridad:** P1
**Puntos :** 2

---

## RF-17 – Inclusión obligatoria de advertencia clínica

**Descripción**
Toda respuesta debe incluir una advertencia indicando que no sustituye el criterio médico profesional.

**Criterios de aceptación**

```gherkin
Feature: Advertencia obligatoria

Scenario: Mostrar advertencia
  DADO que el sistema genera una respuesta
  ENTONCES incluye una advertencia visible indicando que no sustituye el criterio profesional
```

**Prioridad:** P1
**Puntos :** 1

---

## RF-18 – Procesamiento stateless de consultas

**Descripción**
El sistema debe procesar cada consulta de forma independiente, utilizando
únicamente el contenido de la consulta activa para generar la respuesta.

**Criterios de aceptación**
```gherkin
Feature: Procesamiento stateless de consultas

Scenario: Envío de una nueva consulta
  DADO que la usuaria envía una consulta
  CUANDO el sistema la procesa
  ENTONCES genera la respuesta basándose exclusivamente en el contenido
  de la consulta actual
```

**Prioridad:** P1
**Puntos:** 2

---

## RF-19 – Indicador visual de procesamiento

**Descripción**
El sistema debe mostrar un indicador visual mientras procesa la consulta.

**Criterios de aceptación**

```gherkin
Feature: Indicador de procesamiento

Scenario: Consulta en procesamiento
  DADO que la usuaria envía una consulta
  CUANDO el sistema inicia el procesamiento
  ENTONCES muestra un indicador visual
  Y lo oculta cuando la respuesta está disponible
```

**Prioridad:** P1
**Puntos :** 1

---

## RF-20 – Retención automática del historial

**Descripción**
El sistema debe eliminar automáticamente los registros del historial con una antigüedad superior a 180 días (6 meses) mediante una tarea programada en background, sin intervención del usuario.

**Criterios de aceptación**

```gherkin
Feature: Eliminación automática por política de retención

Scenario: Purga periódica de registros vencidos
  DADO que existen registros con antigüedad superior a 180 días
  CUANDO el sistema ejecuta la tarea de purga programada
  ENTONCES elimina todos los registros cuya marca temporal sea anterior al límite de 180 días
  Y no elimina registros dentro del período de retención vigente
  Y no requiere intervención del usuario para ejecutarse
```

**Prioridad:** P1
**Puntos:** 3

