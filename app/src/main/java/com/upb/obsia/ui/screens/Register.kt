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
import com.upb.obsia.data.ChatSession
import com.upb.obsia.data.User
import com.upb.obsia.ui.theme.*
import kotlinx.coroutines.launch

private fun isValidEmail(email: String): Boolean {
        return android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()
}

@Composable
fun RegisterScreen(navController: NavController) {
        val context = LocalContext.current
        val scope = rememberCoroutineScope()
        val db = remember { AppDatabase.getInstance(context) }

        var nombre by remember { mutableStateOf("") }
        var email by remember { mutableStateOf("") }
        var celular by remember { mutableStateOf("") }
        var isLoading by remember { mutableStateOf(false) }

        var emailError by remember { mutableStateOf<String?>(null) }
        var celularError by remember { mutableStateOf<String?>(null) }

        val celularFormatError: String? =
                when {
                        celular.isEmpty() -> null
                        !celular.startsWith("3") -> "El celular debe comenzar por 3"
                        celular.length != 10 -> "El celular debe tener 10 dígitos"
                        else -> null
                }

        val emailFormatError: String? =
                when {
                        email.isEmpty() -> null
                        !isValidEmail(email) -> "Ingresa un correo electrónico válido"
                        else -> null
                }

        val formValid =
                celularFormatError == null &&
                        emailFormatError == null &&
                        celular.isNotEmpty() &&
                        email.isNotEmpty() &&
                        nombre.isNotEmpty()

        Column(modifier = Modifier.fillMaxSize().background(BlancoAuxiliar)) {
                Box(
                        modifier =
                                Modifier.fillMaxWidth()
                                        .clip(
                                                RoundedCornerShape(
                                                        bottomStart = 32.dp,
                                                        bottomEnd = 32.dp
                                                )
                                        )
                                        .background(FondoPrincipal)
                                        .padding(
                                                start = 24.dp,
                                                end = 24.dp,
                                                top = 48.dp,
                                                bottom = 32.dp
                                        )
                ) {
                        Column {
                                IconButton(
                                        onClick = { navController.navigate(NavRoutes.ONBOARDING) },
                                        modifier =
                                                Modifier.size(48.dp)
                                                        .offset(x = (-12).dp, y = (-8).dp)
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
                                        text = "Regístrate",
                                        fontSize = 32.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = FondoBlanco
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                        text = "Llena con tus datos para registrarte",
                                        fontSize = 14.sp,
                                        color = FondoBlanco
                                )
                        }
                }

                Column(
                        modifier =
                                Modifier.fillMaxWidth()
                                        .padding(horizontal = 24.dp, vertical = 32.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                        OutlinedTextField(
                                value = nombre,
                                onValueChange = { nombre = it },
                                label = { Text("Ingresa tu nombre") },
                                placeholder = { Text("Ingresa tu nombre") },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(32.dp),
                                colors =
                                        OutlinedTextFieldDefaults.colors(
                                                focusedBorderColor = FondoPrincipal,
                                                unfocusedBorderColor = FondoPrincipal,
                                                focusedLabelColor = FondoPrincipal,
                                                unfocusedLabelColor = LetrasNegras50,
                                        )
                        )

                        OutlinedTextField(
                                value = email,
                                onValueChange = {
                                        email = it
                                        emailError = null
                                },
                                label = { Text("Ingresa tu correo electrónico") },
                                placeholder = { Text("Ingresa tu correo electrónico") },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(32.dp),
                                keyboardOptions =
                                        KeyboardOptions(keyboardType = KeyboardType.Email),
                                isError = emailFormatError != null || emailError != null,
                                supportingText = {
                                        val msg = emailError ?: emailFormatError
                                        if (msg != null) {
                                                Text(
                                                        text = msg,
                                                        color = MaterialTheme.colorScheme.error
                                                )
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

                        OutlinedTextField(
                                value = celular,
                                onValueChange = {
                                        if (it.length <= 10 && it.all { c -> c.isDigit() }) {
                                                celular = it
                                                celularError = null
                                        }
                                },
                                label = { Text("Ingresa tu número de celular") },
                                placeholder = { Text("Ingresa tu número de celular") },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(32.dp),
                                keyboardOptions =
                                        KeyboardOptions(keyboardType = KeyboardType.Phone),
                                isError = celularFormatError != null || celularError != null,
                                supportingText = {
                                        val msg = celularError ?: celularFormatError
                                        if (msg != null) {
                                                Text(
                                                        text = msg,
                                                        color = MaterialTheme.colorScheme.error
                                                )
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
                        modifier =
                                Modifier.fillMaxWidth()
                                        .padding(horizontal = 24.dp, vertical = 24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                        Button(
                                onClick = {
                                        scope.launch {
                                                isLoading = true
                                                emailError = null
                                                celularError = null
                                                try {
                                                        val porCelular =
                                                                db.userDao().getByCelular(celular)
                                                        val porEmail =
                                                                db.userDao().getByEmail(email)
                                                        when {
                                                                porCelular != null ->
                                                                        celularError =
                                                                                "Este celular ya está registrado"
                                                                porEmail != null ->
                                                                        emailError =
                                                                                "Este correo ya está registrado"
                                                                else -> {
                                                                        // Insertar usuario
                                                                        val newUser =
                                                                                User(
                                                                                        nombre =
                                                                                                nombre,
                                                                                        email =
                                                                                                email,
                                                                                        celular =
                                                                                                celular
                                                                                )
                                                                        db.userDao().insert(newUser)

                                                                        navController
                                                                                .navigate(NavRoutes.LOGIN) {
                                                                                        popUpTo(
                                                                                                NavRoutes
                                                                                                        .REGISTER
                                                                                        ) {
                                                                                                inclusive =
                                                                                                        true
                                                                                        }
                                                                                }
                                                                }
                                                        }
                                                } catch (e: Exception) {
                                                        celularError =
                                                                "Error al registrar. Intenta de nuevo."
                                                } finally {
                                                        isLoading = false
                                                }
                                        }
                                },
                                enabled = formValid && !isLoading,
                                modifier = Modifier.fillMaxWidth().height(56.dp),
                                shape = RoundedCornerShape(32.dp),
                                colors =
                                        ButtonDefaults.buttonColors(containerColor = FondoPrincipal)
                        ) {
                                if (isLoading) {
                                        CircularProgressIndicator(
                                                color = FondoBlanco,
                                                modifier = Modifier.size(24.dp),
                                                strokeWidth = 2.dp
                                        )
                                } else {
                                        Text(
                                                text = "Registrarse",
                                                fontSize = 16.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = FondoBlanco
                                        )
                                }
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                        text = "¿Ya tienes una cuenta? ",
                                        fontSize = 14.sp,
                                        color = LetrasNegras
                                )
                                TextButton(
                                        onClick = { navController.navigate(NavRoutes.LOGIN) },
                                        contentPadding = PaddingValues(0.dp),
                                        modifier = Modifier.height(20.dp)
                                ) {
                                        Text(
                                                text = "Iniciar sesión",
                                                fontSize = 14.sp,
                                                color = FondoPrincipal,
                                                fontWeight = FontWeight.Bold
                                        )
                                }
                        }
                }
        }
}
