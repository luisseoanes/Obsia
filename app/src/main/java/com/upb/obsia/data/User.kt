package com.upb.obsia.data

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
        tableName = "users",
        indices =
                [Index(value = ["celular"], unique = true), Index(value = ["email"], unique = true)]
)
data class User(
        @PrimaryKey(autoGenerate = true) val id: Int = 0,
        val nombre: String,
        val email: String,
        val celular: String
)
