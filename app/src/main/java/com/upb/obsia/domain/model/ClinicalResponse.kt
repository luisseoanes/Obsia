package com.upb.obsia.domain.model

sealed class ClinicalResponse {

    data class Emergency(
        val title: String,
        val why: String,
        val whatsHappening: String = "",
        val dangerLevel: String = "",
        val immediateSteps: List<String>,
        val escalatesWhen: String,
        val urgencyLevel: UrgencyLevel = UrgencyLevel.EMERGENCY,
        val triggeredRuleId: String,
        val disclaimer: String = DISCLAIMER
    ) : ClinicalResponse()

    data class RuleMatch(
        val context: String,
        val response: String,
        val whenToConsult: String,
        val urgencyLevel: UrgencyLevel = UrgencyLevel.ROUTINE,
        val ruleId: String,
        val disclaimer: String = DISCLAIMER
    ) : ClinicalResponse()

    data class LlmGenerated(
        val responseText: String,
        val urgencyLevel: UrgencyLevel,
        val disclaimer: String = DISCLAIMER
    ) : ClinicalResponse()

    data class Error(
        val code: String,
        val message: String
    ) : ClinicalResponse()

    /** Un fragmento de token generado por el LLM en tiempo real. */
    data class StreamToken(val token: String) : ClinicalResponse()

    companion object {
        const val DISCLAIMER =
            "ADVERTENCIA: Esta respuesta no sustituye el criterio médico profesional."
    }
}
