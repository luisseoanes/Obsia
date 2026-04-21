package com.upb.obsia.domain

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class EmergencyRulesConfig(
    val rules: List<EmergencyRule>
)

@Serializable
data class EmergencyRule(
    val id: String,
    val keywords: List<String>,
    val title: String,
    val why: String,
    val whats_happening: String,
    val danger_level: String,
    val immediate_steps: List<String>,
    val escalates_when: String,
    val disclaimer: String
)

data class EmergencyRuleMatch(
    val id: String,
    val title: String,
    val why: String,
    val whats_happening: String,
    val danger_level: String,
    val immediate_steps: List<String>,
    val escalates_when: String,
    val disclaimer: String
)

class EmergencyClinicalRules(jsonContent: String) {
    private val jsonConfig = Json { ignoreUnknownKeys = true }
    private val config = jsonConfig.decodeFromString<EmergencyRulesConfig>(jsonContent)

    fun lookup(query: String): EmergencyRuleMatch? {
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
                    println("LOG: Regla de emergencia [${rule.id}] activada con [$keyword]")
                    return EmergencyRuleMatch(
                        id = rule.id,
                        title = rule.title,
                        why = rule.why,
                        whats_happening = rule.whats_happening,
                        danger_level = rule.danger_level,
                        immediate_steps = rule.immediate_steps,
                        escalates_when = rule.escalates_when,
                        disclaimer = rule.disclaimer
                    )
                }
            }
        }
        return null
    }
}
