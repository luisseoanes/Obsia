package com.upb.obsia.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ChatMessageDao {

    @Insert suspend fun insert(message: ChatMessage): Long

    @Query("SELECT * FROM chat_messages WHERE sessionId = :sessionId ORDER BY createdAt ASC")
    fun getMessagesBySession(sessionId: Int): Flow<List<ChatMessage>>

    @Query("SELECT * FROM chat_messages WHERE sessionId = :sessionId ORDER BY createdAt ASC")
    suspend fun getMessagesBySessionOnce(sessionId: Int): List<ChatMessage>

    @Query("DELETE FROM chat_messages WHERE sessionId = :sessionId")
    suspend fun deleteBySession(sessionId: Int)

    @Query(
            "SELECT content FROM chat_messages WHERE sessionId = :sessionId ORDER BY createdAt DESC LIMIT 1"
    )
    suspend fun getLastMessageBySession(sessionId: Int): String?
}
