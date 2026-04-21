package com.upb.obsia.domain.repository

import com.upb.obsia.domain.model.EngineResponse

/**
 * Contrato del motor de inferencia. El ViewModel depende de esta interfaz, nunca de NativeEngine
 * directamente. Esto permite cambiar el motor (o mockearlo en tests) sin tocar el ViewModel.
 */
interface EngineRepository {

    /**
     * Inicializa el motor de forma lazy. Si ya fue inicializado, retorna inmediatamente sin costo.
     * @return true si el motor quedó listo, false si hubo error.
     */
    suspend fun initialize(): Boolean

    /**
     * Envía una query al motor y retorna la respuesta deserializada completa.
     */
    suspend fun query(text: String): EngineResponse

    /**
     * Envía una query al motor y retorna un flujo de respuestas (tokens en tiempo real).
     */
    fun queryStreaming(text: String): kotlinx.coroutines.flow.Flow<EngineResponse>

    /**
     * Libera los recursos del motor. Llamar cuando la app entra en background prolongado o se
     * destruye.
     */
    fun release()

    /**
     * Retorna true si el motor ya está listo para recibir queries. No realiza ninguna operación —
     * solo consulta estado interno.
     */
    fun isReady(): Boolean
}
