package com.upb.obsia.domain

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class RoutineRulesConfig(
    val rules: List<RoutineRule>
)

@Serializable
data class RoutineRule(
    val id: String,
    val keywords: List<String>,
    val context: String,
    val response: String,
    val when_to_consult: String,
    val disclaimer: String
)

data class RoutineRuleMatch(
    val id: String,
    val context: String,
    val response: String,
    val when_to_consult: String,
    val disclaimer: String
)

class RoutineClinicalRules(jsonContent: String) {
    private val jsonConfig = Json { ignoreUnknownKeys = true }
    private val config = jsonConfig.decodeFromString<RoutineRulesConfig>(jsonContent)

    fun lookup(query: String): RoutineRuleMatch? {
        val textoLimpio = query.lowercase()
            .replace(Regex("[áàä]"), "a")
            .replace(Regex("[éèë]"), "e")
            .replace(Regex("[íìï]"), "i")
            .replace(Regex("[óòö]"), "o")
            .replace(Regex("[úùü]"), "u")
            .replace(Regex("[ñÑ]"), "n")
            .replace(Regex("[^a-z0-9 ]"), "")
            .trim()

        for (rule in config.rules) {
            for (keyword in rule.keywords) {
                if (textoLimpio.contains(keyword)) {
                    println("LOG: Regla de rutina [${rule.id}] activada con [$keyword]")
                    return RoutineRuleMatch(
                        id = rule.id,
                        context = rule.context,
                        response = rule.response,
                        when_to_consult = rule.when_to_consult,
                        disclaimer = rule.disclaimer
                    )
                }
            }
        }
        return null
    }
}
