// Ruta: app/src/main/java/com/upb/obsia/ui/viewmodel/ChatListViewModel.kt

package com.upb.obsia.ui.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.upb.obsia.data.AuthPreferences
import com.upb.obsia.data.ChatSession
import com.upb.obsia.domain.repository.ChatRepository
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class ChatListViewModel(private val context: Context, private val chatRepository: ChatRepository) :
        ViewModel() {

    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()

    private val _sessions = MutableStateFlow<List<ChatSession>>(emptyList())
    val sessions: StateFlow<List<ChatSession>> = _sessions.asStateFlow()

    private val _newSessionId = MutableStateFlow<Int?>(null)
    val newSessionId: StateFlow<Int?> = _newSessionId.asStateFlow()

    private val _selectedSession = MutableStateFlow<ChatSession?>(null)
    val selectedSession: StateFlow<ChatSession?> = _selectedSession.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _lastMessagePreviews = MutableStateFlow<Map<Int, String>>(emptyMap())
    val lastMessagePreviews: StateFlow<Map<Int, String>> = _lastMessagePreviews.asStateFlow()

    private var rawSessions: List<ChatSession> = emptyList()

    @OptIn(ExperimentalCoroutinesApi::class)
    fun loadSessions() {
        val userId = AuthPreferences.getUserId(context)
        if (userId == -1) {
            _errorMessage.value = "No hay sesión activa."
            return
        }

        viewModelScope.launch {
            chatRepository
                    .getSessionsByUser(userId)
                    .catch { e ->
                        _errorMessage.value = e.message ?: "Error al cargar conversaciones."
                    }
                    .collect { list ->
                        rawSessions = list
                        applyFilter(_searchQuery.value)
                        loadLastMessagePreviews(list.map { it.id })
                    }
        }

        viewModelScope.launch { _searchQuery.collect { query -> applyFilter(query) } }
    }

    fun onSearchQueryChange(query: String) {
        _searchQuery.value = query
    }

    fun onSessionLongPress(session: ChatSession) {
        _selectedSession.value = session
    }

    fun dismissRenameDialog() {
        _selectedSession.value = null
    }

    fun renameSession(session: ChatSession, newTitle: String) {
        if (newTitle.isBlank()) return
        viewModelScope.launch {
            try {
                chatRepository.renameSession(session, newTitle)
                _selectedSession.value = null
            } catch (e: Exception) {
                _errorMessage.value = "No se pudo renombrar la conversación."
            }
        }
    }

    fun deleteSession(session: ChatSession) {
        viewModelScope.launch {
            try {
                chatRepository.deleteSession(session.id)
            } catch (e: Exception) {
                _errorMessage.value = "No se pudo eliminar la conversación."
            }
        }
    }

    fun createNewSession() {
        val userId = AuthPreferences.getUserId(context)
        if (userId == -1) {
            _errorMessage.value = "No hay sesión activa."
            return
        }
        viewModelScope.launch {
            try {
                val newSession =
                        ChatSession(
                                userId = userId,
                                title = "Nueva consulta",
                                createdAt = System.currentTimeMillis(),
                                updatedAt = System.currentTimeMillis()
                        )
                val id = chatRepository.createSession(newSession)
                _newSessionId.value = id.toInt()
            } catch (e: Exception) {
                _errorMessage.value = "No se pudo crear la conversación."
            }
        }
    }

    fun onNavigationHandled() {
        _newSessionId.value = null
    }

    fun clearError() {
        _errorMessage.value = null
    }

    private suspend fun loadLastMessagePreviews(sessionIds: List<Int>) {
        val previews = mutableMapOf<Int, String>()
        sessionIds.forEach { sessionId ->
            val last = chatRepository.getLastMessage(sessionId)
            previews[sessionId] =
                    when {
                        last == null -> "¡Arro está aquí para ayudarte!"
                        last.length <= 40 -> last
                        else -> last.take(40) + "…"
                    }
        }
        _lastMessagePreviews.value = previews
    }

    private fun applyFilter(query: String) {
        _sessions.value =
                if (query.isBlank()) {
                    rawSessions
                } else {
                    rawSessions.filter { it.title.contains(query, ignoreCase = true) }
                }
    }
}
