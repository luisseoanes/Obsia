package com.upb.obsia.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update

@Dao
interface UserDao {
    @Insert suspend fun insert(user: User)

    @Update suspend fun update(user: User)

    @Query("SELECT * FROM users WHERE celular = :celular LIMIT 1")
    suspend fun getByCelular(celular: String): User?

    @Query("SELECT * FROM users WHERE email = :email LIMIT 1")
    suspend fun getByEmail(email: String): User?

    @Query("SELECT * FROM users WHERE id = :id LIMIT 1")
    suspend fun getById(id: Int): User?
}