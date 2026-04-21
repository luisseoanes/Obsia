package com.upb.obsia.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
        tableName = "chat_messages",
        foreignKeys =
                [
                        ForeignKey(
                                entity = ChatSession::class,
                                parentColumns = ["id"],
                                childColumns = ["sessionId"],
                                onDelete = ForeignKey.CASCADE
                        )],
        indices = [Index(value = ["sessionId"])]
)
data class ChatMessage(
        @PrimaryKey(autoGenerate = true) val id: Int = 0,
        val sessionId: Int,
        val role: String, // "user" | "assistant"
        val content: String,
        val processingMs: Long = 0L,
        val createdAt: Long = System.currentTimeMillis()
)
