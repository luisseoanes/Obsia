package com.upb.obsia.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
        tableName = "chat_sessions",
        foreignKeys =
                [
                        ForeignKey(
                                entity = User::class,
                                parentColumns = ["id"],
                                childColumns = ["userId"],
                                onDelete = ForeignKey.CASCADE
                        )],
        indices = [Index(value = ["userId"])]
)
data class ChatSession(
        @PrimaryKey(autoGenerate = true) val id: Int = 0,
        val userId: Int,
        val title: String,
        val createdAt: Long = System.currentTimeMillis(),
        val updatedAt: Long = System.currentTimeMillis()
)
