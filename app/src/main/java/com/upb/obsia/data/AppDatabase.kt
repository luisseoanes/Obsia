// Ruta: app/src/main/java/com/upb/obsia/data/AppDatabase.kt

package com.upb.obsia.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [User::class, ChatSession::class, ChatMessage::class], version = 3)
abstract class AppDatabase : RoomDatabase() {

    abstract fun userDao(): UserDao
    abstract fun chatSessionDao(): ChatSessionDao
    abstract fun chatMessageDao(): ChatMessageDao

    companion object {
        @Volatile private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            return INSTANCE
                    ?: synchronized(this) {
                        Room.databaseBuilder(
                                        context.applicationContext,
                                        AppDatabase::class.java,
                                        "obsia_db"
                                )
                                .fallbackToDestructiveMigration()
                                .build()
                                .also { INSTANCE = it }
                    }
        }
    }
}
