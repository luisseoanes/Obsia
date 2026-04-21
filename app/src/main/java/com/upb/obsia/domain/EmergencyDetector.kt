package com.upb.obsia.domain

import com.upb.obsia.domain.model.UrgencyLevel
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class EmergencyConfig(
    val version: String,
    val categories: List<EmergencyCategory>
)

@Serializable
data class EmergencyCategory(
    val id: String,
    val urgency: String,
    val keywords: List<String>
)

class EmergencyDetector(jsonContent: String) {
    private val jsonConfig = Json { ignoreUnknownKeys = true }
    private val config = jsonConfig.decodeFromString<EmergencyConfig>(jsonContent)

    fun analizar(input: String): UrgencyLevel {
        val textoLimpio = input.lowercase()
            .replace(Regex("[áàä]"), "a")
            .replace(Regex("[éèë]"), "e")
            .replace(Regex("[íìï]"), "i")
            .replace(Regex("[óòö]"), "o")
            .replace(Regex("[úùü]"), "u")
            .replace(Regex("[ñÑ]"), "n")
            .replace(Regex("[^a-z0-9 ]"), "")
            .trim()

        for (categoria in config.categories) {
            for (keyword in categoria.keywords) {
                if (textoLimpio.contains(keyword)) {
                    println("LOG: Match en [${categoria.id}] con [$keyword]")
                    return UrgencyLevel.EMERGENCY
                }
            }
        }
        return UrgencyLevel.ROUTINE
    }
}
