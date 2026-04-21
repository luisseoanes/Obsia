package com.upb.obsia.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface ChatSessionDao {

    @Insert suspend fun insert(session: ChatSession): Long

    @Update suspend fun update(session: ChatSession)

    @Query("SELECT * FROM chat_sessions WHERE userId = :userId ORDER BY updatedAt DESC")
    fun getSessionsByUser(userId: Int): Flow<List<ChatSession>>

    @Query("SELECT * FROM chat_sessions WHERE id = :sessionId LIMIT 1")
    suspend fun getById(sessionId: Int): ChatSession?

    @Query("DELETE FROM chat_sessions WHERE id = :sessionId") suspend fun deleteById(sessionId: Int)

    @Query("UPDATE chat_sessions SET title = :title, updatedAt = :updatedAt WHERE id = :sessionId")
    suspend fun renameSession(sessionId: Int, title: String, updatedAt: Long)
}
