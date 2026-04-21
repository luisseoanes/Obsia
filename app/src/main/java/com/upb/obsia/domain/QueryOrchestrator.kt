package com.upb.obsia.domain

import com.upb.obsia.domain.model.ClinicalResponse
import com.upb.obsia.domain.model.UrgencyLevel

class QueryOrchestrator(
    private val emergencyDetector: EmergencyDetector,
    private val emergencyRules: EmergencyClinicalRules,
    private val routineRules: RoutineClinicalRules,
    private val llmEngine: LlmEngine
) {
    suspend fun process(query: String): ClinicalResponse {

        // Paso 1 — Clasificar urgencia
        val urgency = emergencyDetector.analizar(query)

        // Paso 2 — Si es emergencia, buscar regla clínica de emergencia
        if (urgency == UrgencyLevel.EMERGENCY) {
            val emergencyMatch = emergencyRules.lookup(query)
            if (emergencyMatch != null) {
                return ClinicalResponse.Emergency(
                    title           = emergencyMatch.title,
                    why             = emergencyMatch.why,
                    immediateSteps  = emergencyMatch.immediate_steps,
                    escalatesWhen   = emergencyMatch.escalates_when,
                    triggeredRuleId = emergencyMatch.id
                )
            }
            // Emergencia detectada pero sin regla específica
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

        // Paso 4 — Delegar al LLM real (JNI)
        return try {
            val responseText = llmEngine.infer(query)
            ClinicalResponse.LlmGenerated(
                responseText = responseText,
                urgencyLevel = urgency
            )
        } catch (e: Exception) {
            ClinicalResponse.Error("LLM_ERROR", e.message ?: "Error en el motor nativo")
        }
    }
}

interface LlmEngine {
    suspend fun infer(query: String): String
}

/**
 * Implementación real que se conecta con el NativeEngine a través del repositorio.
 */
class NativeLlmEngine(private val repository: com.upb.obsia.domain.repository.EngineRepository) : LlmEngine {
    override suspend fun infer(query: String): String {
        val response = repository.query(query)
        return when (response) {
            is com.upb.obsia.domain.model.EngineResponse.Success -> response.responseText
            is com.upb.obsia.domain.model.EngineResponse.Failure -> throw Exception(response.errorMessage)
            else -> "Respuesta incompleta del motor."
        }
    }
}

class LlmEngineStub : LlmEngine {
    override suspend fun infer(query: String): String =
        "Respuesta generada por LLM (modo stub). ${ClinicalResponse.DISCLAIMER}"
}
