package com.upb.obsia.ui.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.upb.obsia.data.AppDatabase
import com.upb.obsia.data.AuthPreferences
import com.upb.obsia.data.User
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed class EditProfileState {
    object Idle : EditProfileState()
    object Loading : EditProfileState()
    object Success : EditProfileState()
    data class Error(val message: String) : EditProfileState()
}

class EditProfileViewModel : ViewModel() {

    private val _user = MutableStateFlow<User?>(null)
    val user: StateFlow<User?> = _user.asStateFlow()

    private val _state = MutableStateFlow<EditProfileState>(EditProfileState.Idle)
    val state: StateFlow<EditProfileState> = _state.asStateFlow()

    fun loadUser(context: Context) {
        viewModelScope.launch {
            val userId = AuthPreferences.getUserId(context)
            if (userId == -1) return@launch
            val db = AppDatabase.getInstance(context)
            _user.value = withContext(Dispatchers.IO) { db.userDao().getById(userId) }
        }
    }

    fun updateProfile(context: Context, nombre: String, email: String, photoUri: String?) {
        viewModelScope.launch {
            _state.value = EditProfileState.Loading
            try {
                val userId = AuthPreferences.getUserId(context)
                if (userId == -1) {
                    _state.value = EditProfileState.Error("Sesión no válida")
                    return@launch
                }
                val db = AppDatabase.getInstance(context)
                val current = withContext(Dispatchers.IO) { db.userDao().getById(userId) }
                if (current == null) {
                    _state.value = EditProfileState.Error("Usuario no encontrado")
                    return@launch
                }
                val updated = current.copy(nombre = nombre.trim(), email = email.trim())
                withContext(Dispatchers.IO) { db.userDao().update(updated) }
                photoUri?.let { AuthPreferences.savePhotoUri(context, userId, it) }
                _user.value = updated
                _state.value = EditProfileState.Success
            } catch (e: Exception) {
                _state.value = EditProfileState.Error(e.message ?: "Error al actualizar")
            }
        }
    }

    fun resetState() {
        _state.value = EditProfileState.Idle
    }
}