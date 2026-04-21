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

class SettingsViewModel : ViewModel() {

    private val _user = MutableStateFlow<User?>(null)
    val user: StateFlow<User?> = _user.asStateFlow()

    fun loadUser(context: Context) {
        viewModelScope.launch {
            val userId = AuthPreferences.getUserId(context)
            if (userId == -1) return@launch
            val db = AppDatabase.getInstance(context)
            val result = withContext(Dispatchers.IO) { db.userDao().getById(userId) }
            _user.value = result
        }
    }
}
