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

    private fun normalizar(texto: String): String = texto.lowercase()
        .replace(Regex("[áàäâ]"), "a")
        .replace(Regex("[éèëê]"), "e")
        .replace(Regex("[íìïî]"), "i")
        .replace(Regex("[óòöô]"), "o")
        .replace(Regex("[úùüû]"), "u")
        .replace(Regex("[ñÑ]"), "n")
        .replace(Regex("[^a-z0-9 ]"), "")
        .trim()

    // Keywords pre-normalizados al cargar para garantizar match consistente
    private val rulesConKeywordsNormalizados: List<Pair<EmergencyRule, List<String>>> =
        config.rules.map { rule -> rule to rule.keywords.map { normalizar(it) } }

    fun lookup(query: String): EmergencyRuleMatch? {
        val textoLimpio = normalizar(query)
        for ((rule, keywords) in rulesConKeywordsNormalizados) {
            for (keyword in keywords) {
                if (textoLimpio.contains(keyword)) {
                    android.util.Log.d("ObsIA", "EmergencyRule [${rule.id}] con [$keyword]")
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
