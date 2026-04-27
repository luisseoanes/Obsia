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

    private fun normalizar(texto: String): String = texto.lowercase()
        .replace(Regex("[áàäâ]"), "a")
        .replace(Regex("[éèëê]"), "e")
        .replace(Regex("[íìïî]"), "i")
        .replace(Regex("[óòöô]"), "o")
        .replace(Regex("[úùüû]"), "u")
        .replace(Regex("[ñÑ]"), "n")
        .replace(Regex("[^a-z0-9 ]"), "")
        .trim()

    private val rulesConKeywordsNormalizados: List<Pair<RoutineRule, List<String>>> =
        config.rules.map { rule -> rule to rule.keywords.map { normalizar(it) } }

    fun lookup(query: String): RoutineRuleMatch? {
        val textoLimpio = normalizar(query)
        for ((rule, keywords) in rulesConKeywordsNormalizados) {
            for (keyword in keywords) {
                if (textoLimpio.contains(keyword)) {
                    android.util.Log.d("ObsIA", "RoutineRule [${rule.id}] con [$keyword]")
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
