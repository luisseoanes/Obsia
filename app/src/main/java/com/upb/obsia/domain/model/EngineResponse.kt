package com.upb.obsia.domain.model

/**
 * Representa la respuesta del motor nativo ya deserializada. El ViewModel nunca toca JSON — solo
 * consume este modelo.
 */
sealed class EngineResponse {
    data class Success(val responseText: String, val processingMs: Long) : EngineResponse()
    data class Partial(val token: String) : EngineResponse()
    data class Failure(val errorMessage: String) : EngineResponse()
}
