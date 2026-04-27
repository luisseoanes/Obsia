package com.upb.obsia.domain

import com.upb.obsia.domain.model.ClinicalResponse
import com.upb.obsia.domain.model.EngineResponse
import com.upb.obsia.domain.model.UrgencyLevel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

/** Evento emitido por el motor LLM durante generación token a token. */
sealed class LlmStreamEvent {
    data class Token(val text: String) : LlmStreamEvent()
    data class Complete(val fullText: String) : LlmStreamEvent()
    data class Error(val message: String) : LlmStreamEvent()
}

class QueryOrchestrator(
    private val emergencyDetector: EmergencyDetector,
    private val emergencyRules: EmergencyClinicalRules,
    private val routineRules: RoutineClinicalRules,
    private val llmEngine: LlmEngine
) {
    // conversationContext lleva el historial reciente; query es solo la consulta actual para el matching de reglas
    suspend fun process(query: String, conversationContext: String = query): ClinicalResponse {

        // Paso 1 — Clasificar urgencia solo con la consulta actual
        val urgency = emergencyDetector.analizar(query)

        // Paso 2 — Si es emergencia, buscar regla clínica de emergencia
        if (urgency == UrgencyLevel.EMERGENCY) {
            val emergencyMatch = emergencyRules.lookup(query)
            if (emergencyMatch != null) {
                return ClinicalResponse.Emergency(
                    title           = emergencyMatch.title,
                    why             = emergencyMatch.why,
                    whatsHappening  = emergencyMatch.whats_happening,
                    dangerLevel     = emergencyMatch.danger_level,
                    immediateSteps  = emergencyMatch.immediate_steps,
                    escalatesWhen   = emergencyMatch.escalates_when,
                    triggeredRuleId = emergencyMatch.id
                )
            }
            return ClinicalResponse.Emergency(
                title           = "Emergencia obstétrica detectada",
                why             = "Se detectaron indicadores de emergencia en la consulta.",
                immediateSteps  = listOf("Llame al 123 inmediatamente."),
                escalatesWhen   = "En cualquier momento sin atención médica.",
                triggeredRuleId = "GENERIC"
            )
        }

        // Paso 3 — Buscar en reglas de rutina
        val routineMatch = routineRules.lookup(query)
        if (routineMatch != null) {
            return ClinicalResponse.RuleMatch(
                context       = routineMatch.context,
                response      = routineMatch.response,
                whenToConsult = routineMatch.when_to_consult,
                ruleId        = routineMatch.id
            )
        }

        // Paso 4 — Delegar al LLM con contexto completo de conversación
        return try {
            val responseText = llmEngine.infer(conversationContext)
            ClinicalResponse.LlmGenerated(
                responseText = responseText,
                urgencyLevel = urgency
            )
        } catch (e: Exception) {
            ClinicalResponse.Error("LLM_ERROR", e.message ?: "Error en el motor nativo")
        }
    }

    /**
     * Versión streaming de [process]. Para rutas de reglas emite un solo item y cierra.
     * Para la ruta LLM emite [ClinicalResponse.StreamToken] por cada token y
     * termina con [ClinicalResponse.LlmGenerated] que contiene el texto final validado.
     */
    fun processStreaming(query: String, conversationContext: String = query): Flow<ClinicalResponse> = flow {
        val urgency = emergencyDetector.analizar(query)

        if (urgency == UrgencyLevel.EMERGENCY) {
            val emergencyMatch = emergencyRules.lookup(query)
            if (emergencyMatch != null) {
                emit(ClinicalResponse.Emergency(
                    title           = emergencyMatch.title,
                    why             = emergencyMatch.why,
                    whatsHappening  = emergencyMatch.whats_happening,
                    dangerLevel     = emergencyMatch.danger_level,
                    immediateSteps  = emergencyMatch.immediate_steps,
                    escalatesWhen   = emergencyMatch.escalates_when,
                    triggeredRuleId = emergencyMatch.id
                ))
            } else {
                emit(ClinicalResponse.Emergency(
                    title           = "Emergencia obstétrica detectada",
                    why             = "Se detectaron indicadores de emergencia en la consulta.",
                    immediateSteps  = listOf("Llame al 123 inmediatamente."),
                    escalatesWhen   = "En cualquier momento sin atención médica.",
                    triggeredRuleId = "GENERIC"
                ))
            }
            return@flow
        }

        val routineMatch = routineRules.lookup(query)
        if (routineMatch != null) {
            emit(ClinicalResponse.RuleMatch(
                context       = routineMatch.context,
                response      = routineMatch.response,
                whenToConsult = routineMatch.when_to_consult,
                ruleId        = routineMatch.id
            ))
            return@flow
        }

        // Ruta LLM: emitir tokens uno a uno, luego el resultado final validado
        llmEngine.inferStreaming(conversationContext).collect { event ->
            when (event) {
                is LlmStreamEvent.Token    -> emit(ClinicalResponse.StreamToken(event.text))
                is LlmStreamEvent.Complete -> emit(ClinicalResponse.LlmGenerated(
                    responseText = event.fullText,
                    urgencyLevel = urgency
                ))
                is LlmStreamEvent.Error    -> emit(ClinicalResponse.Error("LLM_ERROR", event.message))
            }
        }
    }
}

interface LlmEngine {
    suspend fun infer(query: String): String
    fun inferStreaming(query: String): Flow<LlmStreamEvent>
}

/**
 * Implementación real que se conecta con el NativeEngine a través del repositorio.
 */
class NativeLlmEngine(private val repository: com.upb.obsia.domain.repository.EngineRepository) : LlmEngine {
    override suspend fun infer(query: String): String {
        val response = repository.query(query)
        return when (response) {
            is EngineResponse.Success -> response.responseText
            is EngineResponse.Failure -> throw Exception(response.errorMessage)
            else -> "Respuesta incompleta del motor."
        }
    }

    override fun inferStreaming(query: String): Flow<LlmStreamEvent> = flow {
        repository.queryStreaming(query).collect { response ->
            when (response) {
                is EngineResponse.Partial  -> emit(LlmStreamEvent.Token(response.token))
                is EngineResponse.Success  -> emit(LlmStreamEvent.Complete(response.responseText))
                is EngineResponse.Failure  -> emit(LlmStreamEvent.Error(response.errorMessage))
            }
        }
    }.flowOn(Dispatchers.IO)
}

class LlmEngineStub : LlmEngine {
    override suspend fun infer(query: String): String =
        "Respuesta generada por LLM (modo stub). ${ClinicalResponse.DISCLAIMER}"

    override fun inferStreaming(query: String): Flow<LlmStreamEvent> = flow {
        val full = "Respuesta generada por LLM (modo stub). ${ClinicalResponse.DISCLAIMER}"
        full.chunked(3).forEach { chunk -> emit(LlmStreamEvent.Token(chunk)) }
        emit(LlmStreamEvent.Complete(full))
    }
}
