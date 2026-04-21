// Ruta: app/src/main/java/com/upb/obsia/data/repository/ChatRepositoryImpl.kt

package com.upb.obsia.data.repository

import com.upb.obsia.data.ChatMessage
import com.upb.obsia.data.ChatMessageDao
import com.upb.obsia.data.ChatSession
import com.upb.obsia.data.ChatSessionDao
import com.upb.obsia.domain.repository.ChatRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext

class ChatRepositoryImpl(
        private val chatMessageDao: ChatMessageDao,
        private val chatSessionDao: ChatSessionDao
) : ChatRepository {

    // ─── ChatScreen ───────────────────────────────────────────────────────────

    override suspend fun getSessionName(sessionId: Int): String? =
            withContext(Dispatchers.IO) { chatSessionDao.getById(sessionId)?.title }

    override suspend fun getMessages(sessionId: Int): List<ChatMessage> =
            withContext(Dispatchers.IO) { chatMessageDao.getMessagesBySessionOnce(sessionId) }

    override suspend fun saveMessage(message: ChatMessage): Long =
            withContext(Dispatchers.IO) { chatMessageDao.insert(message) }

    override suspend fun touchSession(sessionId: Int) {
        withContext(Dispatchers.IO) {
            chatSessionDao.getById(sessionId)?.let { session ->
                chatSessionDao.update(session.copy(updatedAt = System.currentTimeMillis()))
            }
        }
    }

    // ─── ChatList ─────────────────────────────────────────────────────────────

    override fun getSessionsByUser(userId: Int): Flow<List<ChatSession>> =
            // Flow de Room ya es seguro para hilos — no envolver en withContext,
            // hacerlo en un Flow frío rompe la reactividad.
            chatSessionDao.getSessionsByUser(userId)

    override suspend fun createSession(session: ChatSession): Long =
            withContext(Dispatchers.IO) { chatSessionDao.insert(session) }

    override suspend fun renameSession(session: ChatSession, newTitle: String) {
        withContext(Dispatchers.IO) {
            chatSessionDao.update(
                    session.copy(title = newTitle.trim(), updatedAt = System.currentTimeMillis())
            )
        }
    }

    override suspend fun deleteSession(sessionId: Int) {
        withContext(Dispatchers.IO) { chatSessionDao.deleteById(sessionId) }
    }

    override suspend fun getLastMessage(sessionId: Int): String? =
            withContext(Dispatchers.IO) { chatMessageDao.getLastMessageBySession(sessionId) }
}
