// Ruta: app/src/main/java/com/upb/obsia/ui/viewmodel/ChatViewModel.kt

package com.upb.obsia.ui.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.upb.obsia.data.AuthPreferences
import com.upb.obsia.data.ChatMessage
import com.upb.obsia.domain.model.EngineResponse
import com.upb.obsia.domain.repository.ChatRepository
import com.upb.obsia.domain.repository.EngineRepository
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService

sealed class ChatInitState {
    object CopyingAssets : ChatInitState()
    object Initializing : ChatInitState()
    object Ready : ChatInitState()
    data class Error(val message: String) : ChatInitState()
}

sealed class ChatQueryState {
    object Idle : ChatQueryState()
    object Loading : ChatQueryState()
    data class Error(val message: String) : ChatQueryState()
}

sealed class VoiceState {
    object Idle : VoiceState()
    object Listening : VoiceState()
    data class Error(val message: String) : VoiceState()
}

class ChatViewModel(
        private val context: Context,
        private val engineRepository: EngineRepository,
        private val chatRepository: ChatRepository,
        private val queryOrchestrator: com.upb.obsia.domain.QueryOrchestrator
) : ViewModel() {

    private val _initState = MutableStateFlow<ChatInitState>(ChatInitState.CopyingAssets)
    val initState: StateFlow<ChatInitState> = _initState.asStateFlow()

    private val _queryState = MutableStateFlow<ChatQueryState>(ChatQueryState.Idle)
    val queryState: StateFlow<ChatQueryState> = _queryState.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _sessionName = MutableStateFlow("Chat")
    val sessionName: StateFlow<String> = _sessionName.asStateFlow()

    private val _voiceState = MutableStateFlow<VoiceState>(VoiceState.Idle)
    val voiceState: StateFlow<VoiceState> = _voiceState.asStateFlow()

    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    val welcomeMessage = "¡Arro está aquí para ayudarte!"

    private var sessionId: Int = -1
    private var voskModel: Model? = null
    private var speechService: SpeechService? = null

    fun initialize(sessionId: Int) {
        if (_initState.value is ChatInitState.Ready) return

        this.sessionId = sessionId

        val userId = AuthPreferences.getUserId(context)
        if (userId == -1) {
            _initState.value = ChatInitState.Error("No hay sesión de usuario activa.")
            return
        }

        viewModelScope.launch {
            _sessionName.value = chatRepository.getSessionName(sessionId) ?: "Chat"
            _initState.value = ChatInitState.Initializing

            val engineReady = engineRepository.initialize()
            if (!engineReady) {
                _initState.value = ChatInitState.Error("El motor no pudo inicializarse.")
                return@launch
            }

            _messages.value = chatRepository.getMessages(sessionId)
            _initState.value = ChatInitState.Ready

            // NOTA: Vosk ya no se inicializa aquí, sino de forma lazy en toggleVoice
        }
    }

    private suspend fun initVoskModel() =
            withContext(Dispatchers.IO) {
                try {
                    val modelDir = File(context.filesDir, "vosk-model-small-es-0.42")
                    if (!modelDir.exists()) {
                        copyModelFromAssets(modelDir)
                    }
                    voskModel = Model(modelDir.absolutePath)
                } catch (e: Exception) {
                    _voiceState.value = VoiceState.Error("Modelo de voz no disponible")
                }
            }

    private fun copyModelFromAssets(destDir: File) {
        destDir.mkdirs()
        copyAssetFolder("vosk-model-small-es-0.42", destDir)
    }

    private fun copyAssetFolder(assetPath: String, destDir: File) {
        val assets = context.assets.list(assetPath) ?: return
        if (assets.isEmpty()) {
            context.assets.open(assetPath).use { input ->
                destDir.outputStream().use { output -> input.copyTo(output) }
            }
        } else {
            destDir.mkdirs()
            assets.forEach { child -> copyAssetFolder("$assetPath/$child", File(destDir, child)) }
        }
    }

    fun toggleVoice() {
        viewModelScope.launch {
            if (voskModel == null) {
                _voiceState.value = VoiceState.Listening // Estado visual de carga
                initVoskModel()
            }

            when (_voiceState.value) {
                is VoiceState.Listening -> {
                    if (speechService != null) stopListening()
                    else startListening()
                }
                else -> startListening()
            }
        }
    }

    private fun startListening() {
        val model = voskModel
        if (model == null) {
            _voiceState.value = VoiceState.Error("Modelo de voz no cargado")
            return
        }

        try {
            val recognizer = Recognizer(model, 16000.0f)
            speechService =
                    SpeechService(recognizer, 16000.0f).apply {
                        startListening(
                                object : RecognitionListener {
                                    override fun onPartialResult(hypothesis: String?) {}

                                    override fun onResult(hypothesis: String?) {
                                        hypothesis ?: return
                                        val result = parseVoskResult(hypothesis, "text")
                                        if (result.isNotBlank()) {
                                            val current = _inputText.value
                                            _inputText.value =
                                                    if (current.isBlank()) result
                                                    else "$current $result"
                                        }
                                    }

                                    override fun onFinalResult(hypothesis: String?) {
                                        hypothesis ?: return
                                        val result = parseVoskResult(hypothesis, "text")
                                        if (result.isNotBlank()) {
                                            val current = _inputText.value
                                            _inputText.value =
                                                    if (current.isBlank()) result
                                                    else "$current $result"
                                        }
                                        _voiceState.value = VoiceState.Idle
                                    }

                                    override fun onError(exception: Exception?) {
                                        _voiceState.value =
                                                VoiceState.Error(
                                                        exception?.message
                                                                ?: "Error de reconocimiento"
                                                )
                                    }

                                    override fun onTimeout() {
                                        stopListening()
                                    }
                                }
                        )
                    }
            _voiceState.value = VoiceState.Listening
        } catch (e: Exception) {
            _voiceState.value = VoiceState.Error("No se pudo iniciar el micrófono")
        }
    }

    private fun stopListening() {
        speechService?.stop()
        speechService?.shutdown()
        speechService = null
        _voiceState.value = VoiceState.Idle
    }

    private fun parseVoskResult(hypothesis: String, key: String): String {
        return try {
            JSONObject(hypothesis).optString(key, "").trim()
        } catch (e: Exception) {
            ""
        }
    }

    fun updateInputText(text: String) {
        _inputText.value = text
    }

    fun clearInputText() {
        _inputText.value = ""
    }

    fun sendMessage(text: String) {
        if (_initState.value !is ChatInitState.Ready) return
        if (_queryState.value is ChatQueryState.Loading) return
        if (text.isBlank()) return

        if (_voiceState.value is VoiceState.Listening) stopListening()

        viewModelScope.launch {
            val userMessage =
                    ChatMessage(sessionId = sessionId, role = "user", content = text.trim())
            val userMsgId = chatRepository.saveMessage(userMessage)
            _messages.value = _messages.value + userMessage.copy(id = userMsgId.toInt())

            chatRepository.touchSession(sessionId)
            _queryState.value = ChatQueryState.Loading

            // Orquestar consulta — Esto detecta emergencias y reglas clínicas antes del LLM
            val response = queryOrchestrator.process(text.trim())
            handleClinicalResponse(response)
        }
    }

    private suspend fun handleClinicalResponse(response: com.upb.obsia.domain.model.ClinicalResponse) {
        when (response) {
            is com.upb.obsia.domain.model.ClinicalResponse.Emergency -> {
                val content = buildString {
                    append("**${response.title}**\n\n")
                    append("${response.why}\n\n")
                    append("### Acciones inmediatas:\n")
                    response.immediateSteps.forEach { append("* $it\n") }
                    append("\n⚠️ ${response.disclaimer}")
                }
                saveAndShowAssistantMessage(content)
            }
            is com.upb.obsia.domain.model.ClinicalResponse.RuleMatch -> {
                val content = buildString {
                    append("${response.response}\n\n")
                    append("*¿Cuándo consultar?* ${response.whenToConsult}\n\n")
                    append("ℹ️ ${response.disclaimer}")
                }
                saveAndShowAssistantMessage(content)
            }
            is com.upb.obsia.domain.model.ClinicalResponse.LlmGenerated -> {
                // Si es LLM, preferimos streaming para mejor UX, pero usamos el texto del orquestador
                // como fallback o si el streaming falla. Aquí reiniciamos el streaming.
                startStreamingResponse(_messages.value.last().content)
            }
            is com.upb.obsia.domain.model.ClinicalResponse.Error -> {
                _queryState.value = ChatQueryState.Error(response.message)
            }
        }
    }

    private suspend fun startStreamingResponse(text: String) {
        var currentAssistantMsg =
                ChatMessage(sessionId = sessionId, role = "assistant", content = "")

        engineRepository.queryStreaming(text.trim()).collect { response ->
            when (response) {
                is EngineResponse.Partial -> {
                    _queryState.value = ChatQueryState.Idle
                    currentAssistantMsg =
                            currentAssistantMsg.copy(
                                    content = currentAssistantMsg.content + response.token
                            )
                    updateLastAssistantMessage(currentAssistantMsg)
                }
                is EngineResponse.Success -> {
                    val finalMsg =
                            currentAssistantMsg.copy(
                                    content = response.responseText,
                                    processingMs = response.processingMs
                            )
                    val id = chatRepository.saveMessage(finalMsg)
                    updateLastAssistantMessage(finalMsg.copy(id = id.toInt()))
                    _queryState.value = ChatQueryState.Idle
                }
                is EngineResponse.Failure -> {
                    _queryState.value = ChatQueryState.Error(response.errorMessage)
                }
            }
        }
    }

    private suspend fun saveAndShowAssistantMessage(content: String) {
        val assistantMessage =
                ChatMessage(sessionId = sessionId, role = "assistant", content = content)
        val id = chatRepository.saveMessage(assistantMessage)
        _messages.value = _messages.value + assistantMessage.copy(id = id.toInt())
        _queryState.value = ChatQueryState.Idle
    }

    private fun updateLastAssistantMessage(msg: ChatMessage) {
        val lastMsg = _messages.value.lastOrNull()
        if (lastMsg?.role == "assistant") {
            _messages.value = _messages.value.dropLast(1) + msg
        } else {
            _messages.value = _messages.value + msg
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopListening()
        voskModel?.close()
        engineRepository.release() // Liberar RAM del LLM
    }
}
