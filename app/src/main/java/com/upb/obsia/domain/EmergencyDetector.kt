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

    private fun normalizar(texto: String): String = texto.lowercase()
        .replace(Regex("[áàäâ]"), "a")
        .replace(Regex("[éèëê]"), "e")
        .replace(Regex("[íìïî]"), "i")
        .replace(Regex("[óòöô]"), "o")
        .replace(Regex("[úùüû]"), "u")
        .replace(Regex("[ñÑ]"), "n")
        .replace(Regex("[^a-z0-9 ]"), "")
        .trim()

    // Keywords pre-normalizados al cargar para evitar mismatch con texto normalizado
    private val keywordsPorCategoria: List<Pair<String, List<String>>> =
        config.categories.map { cat -> cat.id to cat.keywords.map { normalizar(it) } }

    fun analizar(input: String): UrgencyLevel {
        val textoLimpio = normalizar(input)
        for ((categoriaId, keywords) in keywordsPorCategoria) {
            for (keyword in keywords) {
                if (textoLimpio.contains(keyword)) {
                    android.util.Log.d("ObsIA", "EmergencyMatch [$categoriaId] con [$keyword]")
                    return UrgencyLevel.EMERGENCY
                }
            }
        }
        return UrgencyLevel.ROUTINE
    }
}
