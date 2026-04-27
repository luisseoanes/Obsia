package com.upb.obsia.data.repository

import android.content.Context
import com.obsIA.engine.NativeEngine
import com.upb.obsia.domain.model.EngineResponse
import com.upb.obsia.domain.repository.EngineRepository
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject

class EngineRepositoryImpl(private val context: Context) : EngineRepository {

    private val engine = NativeEngine()
    private val initMutex = Mutex()
    private var initialized = false

    // Optimización: Detectar núcleos para hilos nativos
    private val optimalThreads =
            (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)

    override suspend fun initialize(): Boolean {
        if (initialized) return true

        return initMutex.withLock {
            if (initialized) return@withLock true

            try {
                val modelFile = copyAssetIfNeeded("qwen2.5-3b-instruct-q4_k_m.gguf")
                val chunksFile = copyAssetIfNeeded("chunks.json")

                val result =
                        withContext(Dispatchers.IO) {
                            engine.init(
                                    modelPath = modelFile.absolutePath,
                                    ragPath = chunksFile.absolutePath,
                                    nThreads = optimalThreads
                            )
                        }

                initialized = (result == 0)
                initialized
            } catch (e: Exception) {
                initialized = false
                false
            }
        }
    }

    override suspend fun query(text: String): EngineResponse {
        return try {
            val rawJson = withContext(Dispatchers.IO) { engine.processQuery(text) }
            parseResponse(rawJson)
        } catch (e: Exception) {
            EngineResponse.Failure(e.message ?: "Error desconocido en el motor")
        }
    }

    override fun queryStreaming(text: String): Flow<EngineResponse> = callbackFlow {
        if (!initialized) {
            trySend(EngineResponse.Failure("Motor no inicializado"))
            close()
            return@callbackFlow
        }

        val callback =
                object : NativeEngine.TokenCallback {
                    override fun onToken(token: String): Boolean {
                        trySend(EngineResponse.Partial(token))
                        return true // Continuar generación
                    }
                }

        val rawJson = engine.processQueryStreaming(text, callback)
        trySend(parseResponse(rawJson))
        close()

        awaitClose {
            // El motor nativo no tiene "cancel" por query individual, 
            // pero podríamos manejar estados de interrupción si fuera necesario.
        }
    }.flowOn(Dispatchers.IO)

    override fun release() {
        if (initialized) {
            engine.release()
            initialized = false
        }
    }

    override fun isReady(): Boolean = initialized

    private suspend fun copyAssetIfNeeded(assetName: String): File =
            withContext(Dispatchers.IO) {
                val dest = File(context.filesDir, assetName)
                val shouldCopy = if (!dest.exists()) {
                    true
                } else {
                    // Reemplazar si el asset del APK es más grande o más nuevo que el cacheado.
                    // Esto garantiza que actualizaciones de JSON lleguen a usuarios existentes.
                    val assetSize = context.assets.openFd(assetName).use { it.length }
                    dest.length() != assetSize
                }
                if (shouldCopy) {
                    context.assets.open(assetName).use { input ->
                        FileOutputStream(dest).use { output -> input.copyTo(output) }
                    }
                }
                dest
            }

    private fun parseResponse(rawJson: String): EngineResponse {
        return try {
            val json = JSONObject(rawJson)
            when (json.optString("status")) {
                "ok" ->
                        EngineResponse.Success(
                                responseText = json.optString("response_text"),
                                processingMs = json.optLong("processing_ms")
                        )
                else ->
                        EngineResponse.Failure(
                                json.optString("error_message", "Error al procesar la consulta")
                        )
            }
        } catch (e: Exception) {
            EngineResponse.Failure("Respuesta del motor con formato inválido")
        }
    }
}
