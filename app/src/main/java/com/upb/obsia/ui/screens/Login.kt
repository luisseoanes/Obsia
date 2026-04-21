package com.upb.obsia.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.upb.obsia.NavRoutes
import com.upb.obsia.data.AppDatabase
import com.upb.obsia.data.AuthPreferences
import com.upb.obsia.ui.theme.*
import kotlinx.coroutines.launch

@Composable
fun LoginScreen(navController: NavController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val db = remember { AppDatabase.getInstance(context) }

    var celular by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var celularError by remember { mutableStateOf<String?>(null) }

    val celularFormatError: String? =
            when {
                celular.isEmpty() -> null
                !celular.startsWith("3") -> "El celular debe comenzar por 3"
                celular.length != 10 -> "El celular debe tener 10 dígitos"
                else -> null
            }

    val formValid = celularFormatError == null && celular.isNotEmpty()

    Column(modifier = Modifier.fillMaxSize().background(BlancoAuxiliar)) {
        Box(
                modifier =
                        Modifier.fillMaxWidth()
                                .clip(RoundedCornerShape(bottomStart = 32.dp, bottomEnd = 32.dp))
                                .background(FondoPrincipal)
                                .padding(start = 24.dp, end = 24.dp, top = 48.dp, bottom = 32.dp)
        ) {
            Column {
                IconButton(
                        onClick = {
                            navController.navigate(NavRoutes.ONBOARDING) {
                                popUpTo(NavRoutes.LOGIN) { inclusive = true }
                            }
                        },
                        modifier = Modifier.size(48.dp).offset(x = (-12).dp, y = (-8).dp)
                ) {
                    Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Volver",
                            tint = FondoBlanco,
                            modifier = Modifier.size(38.dp)
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                        text = "Iniciar Sesión",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        color = FondoBlanco
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                        text = "Ingresa tu número de 10 dígitos para continuar.",
                        fontSize = 14.sp,
                        color = FondoBlanco
                )
            }
        }

        Column(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 32.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            OutlinedTextField(
                    value = celular,
                    onValueChange = {
                        if (it.length <= 10 && it.all { c -> c.isDigit() }) {
                            celular = it
                            celularError = null
                        }
                    },
                    label = { Text("Celular") },
                    placeholder = { Text("Ingresa tu número") },
                    prefix = { Text("+57  ") },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(32.dp),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                    isError = celularFormatError != null || celularError != null,
                    supportingText = {
                        val msg = celularError ?: celularFormatError
                        if (msg != null) {
                            Text(text = msg, color = MaterialTheme.colorScheme.error)
                        }
                    },
                    colors =
                            OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = FondoPrincipal,
                                    unfocusedBorderColor = FondoPrincipal,
                                    focusedLabelColor = FondoPrincipal,
                                    unfocusedLabelColor = LetrasNegras50,
                            )
            )
        }

        Spacer(modifier = Modifier.weight(1f))

        Column(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Button(
                    onClick = {
                        scope.launch {
                            isLoading = true
                            celularError = null
                            try {
                                val user = db.userDao().getByCelular(celular)
                                if (user == null) {
                                    celularError = "No existe una cuenta con este número"
                                } else {
                                    AuthPreferences.saveUserId(context, user.id)
                                    navController.navigate(NavRoutes.CHAT_LIST) {
                                        popUpTo(NavRoutes.LOGIN) { inclusive = true }
                                    }
                                }
                            } catch (e: Exception) {
                                celularError = "Error al iniciar sesión. Intenta de nuevo."
                            } finally {
                                isLoading = false
                            }
                        }
                    },
                    enabled = formValid && !isLoading,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = FondoPrincipal)
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                            color = FondoBlanco,
                            modifier = Modifier.size(24.dp),
                            strokeWidth = 2.dp
                    )
                } else {
                    Text(
                            text = "Iniciar Sesión",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            color = FondoBlanco
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(text = "¿No tienes una cuenta? ", fontSize = 14.sp, color = LetrasNegras)
                TextButton(
                        onClick = {
                            navController.navigate(NavRoutes.REGISTER) {
                                popUpTo(NavRoutes.LOGIN) { inclusive = true }
                            }
                        },
                        contentPadding = PaddingValues(0.dp),
                        modifier = Modifier.height(20.dp)
                ) {
                    Text(
                            text = "Regístrate",
                            fontSize = 14.sp,
                            color = FondoPrincipal,
                            fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}
