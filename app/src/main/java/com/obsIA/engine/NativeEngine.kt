// Ruta: app/src/main/java/com/obsIA/engine/NativeEngine.kt
package com.obsIA.engine

/**
 * Android Native Bridge para el motor médico Obsia. El package com.obsIA.engine debe coincidir
 * exactamente con las firmas JNI del .so compilado. NO cambiar el package name sin recompilar la
 * librería nativa.
 */
class NativeEngine {

    companion object {
        init {
            // Carga 'libobsia_jni.so' desde el APK. El nombre debe coincidir con
            // el target_link_libraries del CMakeLists.txt (sin prefijo "lib" ni extensión ".so").
            System.loadLibrary("obsia_jni")
        }
    }

    /**
     * Interfaz de callback para recibir tokens en tiempo real durante la generación. Implementar
     * esta interfaz para procesar cada token a medida que se genera.
     *
     * IMPORTANTE: El callback se invoca desde el hilo nativo. No actualizar UI directamente aquí —
     * usar Handler o StateFlow para propagar al hilo principal si es necesario.
     */
    interface TokenCallback {
        /**
         * Llamado por cada fragmento de token generado.
         * @param token El fragmento de texto recién generado.
         * @return true para continuar la generación, false para detenerla anticipadamente.
         */
        fun onToken(token: String): Boolean
    }

    /**
     * Inicializa el motor.
     * @param modelPath Ruta absoluta al archivo .gguf en el filesystem del dispositivo.
     * @param ragPath Ruta absoluta al archivo chunks.json en el filesystem del dispositivo.
     * @param nThreads Número de hilos (0 = auto-detectar óptimo, 2-4 recomendado para mobile).
     * @return 0 en éxito, valor negativo en error.
     */
    external fun init(modelPath: String, ragPath: String, nThreads: Int): Int

    /**
     * Envía un mensaje al asistente médico. BLOQUEANTE — llamar siempre desde Dispatchers.IO.
     * @param query Síntomas o pregunta del usuario.
     * @return JSON string con campos: "status", "response_text", "processing_ms".
     */
    external fun processQuery(query: String): String

    /**
     * Envía un mensaje con generación token por token. Los tokens llegan en tiempo real a través
     * del callback. BLOQUEANTE — llamar siempre desde Dispatchers.IO.
     *
     * @param query Síntomas o pregunta del usuario.
     * @param callback Recibe cada token a medida que se genera. Puede ser null para ignorar
     * ```
     *                 el streaming y obtener solo el resultado final.
     * @return
     * ```
     * JSON string con campos finales: "status", "response_text", "processing_ms".
     */
    external fun processQueryStreaming(query: String, callback: TokenCallback?): String

    /** Verifica si el motor está listo para recibir queries. */
    external fun isReady(): Boolean

    /** Libera memoria y recursos del modelo. */
    external fun release()

    /** Retorna el uso actual de memoria del proceso en MB. */
    external fun getMemoryUsageMB(): Int

    /** 
     * Guarda el estado actual del contexto (KV Cache) en un archivo. 
     * Útil para retomar conversaciones sin procesar todo de nuevo.
     */
    external fun saveSession(path: String): Int

    /** 
     * Carga un estado de contexto previamente guardado.
     */
    external fun loadSession(path: String): Int
}
