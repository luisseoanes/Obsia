// Ruta: app/src/main/java/com/upb/obsia/domain/repository/ChatRepository.kt

package com.upb.obsia.domain.repository

import com.upb.obsia.data.ChatMessage
import com.upb.obsia.data.ChatSession
import kotlinx.coroutines.flow.Flow

interface ChatRepository {

    // ─── Operaciones de ChatScreen ────────────────────────────────────────────

    /** Retorna el nombre de la sesión, o null si no existe. */
    suspend fun getSessionName(sessionId: Int): String?

    /** Carga el historial de mensajes de una sesión, ordenados por fecha. */
    suspend fun getMessages(sessionId: Int): List<ChatMessage>

    /** Persiste un mensaje y retorna su id generado. */
    suspend fun saveMessage(message: ChatMessage): Long

    /**
     * Actualiza el timestamp de última actividad de la sesión. Se llama después de cada mensaje
     * para mantener el orden en ChatList.
     */
    suspend fun touchSession(sessionId: Int)

    // ─── Operaciones de ChatList ──────────────────────────────────────────────

    /** Flow reactivo de sesiones de un usuario, ordenadas por última actividad. */
    fun getSessionsByUser(userId: Int): Flow<List<ChatSession>>

    /** Crea una nueva sesión y retorna su id generado. */
    suspend fun createSession(session: ChatSession): Long

    /** Renombra una sesión existente. */
    suspend fun renameSession(session: ChatSession, newTitle: String)

    /** Elimina una sesión por id. */
    suspend fun deleteSession(sessionId: Int)

    /** Retorna el último mensaje de una sesión para el preview, o null si no hay mensajes. */
    suspend fun getLastMessage(sessionId: Int): String?
}
