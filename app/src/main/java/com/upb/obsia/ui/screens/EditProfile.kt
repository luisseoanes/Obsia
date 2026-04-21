package com.upb.obsia.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import coil.compose.AsyncImage
import com.upb.obsia.data.AuthPreferences
import com.upb.obsia.ui.theme.*
import com.upb.obsia.ui.viewmodel.EditProfileState
import com.upb.obsia.ui.viewmodel.EditProfileViewModel

@Composable
fun EditProfileScreen(
    navController: NavController,
    editProfileViewModel: EditProfileViewModel = viewModel()
) {
    val context = LocalContext.current
    val user by editProfileViewModel.user.collectAsState()
    val state by editProfileViewModel.state.collectAsState()
    val snackbarHostState = remember { SnackbarHostState() }

    var nombre by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var photoUri by remember { mutableStateOf<Uri?>(null) }

    val userId = AuthPreferences.getUserId(context)
    val savedPhotoUri = AuthPreferences.getPhotoUri(context, userId)

    val emailFormatError: String? = when {
        email.isEmpty() -> null
        !android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches() -> "Correo no válido"
        else -> null
    }

    val formValid = nombre.isNotEmpty() && email.isNotEmpty() && emailFormatError == null

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        photoUri = uri
    }

    LaunchedEffect(Unit) {
        editProfileViewModel.loadUser(context)
    }

    LaunchedEffect(user) {
        user?.let {
            nombre = it.nombre
            email = it.email
        }
    }

    LaunchedEffect(state) {
        when (state) {
            is EditProfileState.Success -> {
                snackbarHostState.showSnackbar("Perfil actualizado correctamente")
                editProfileViewModel.resetState()
                navController.popBackStack()
            }
            is EditProfileState.Error -> {
                snackbarHostState.showSnackbar((state as EditProfileState.Error).message)
                editProfileViewModel.resetState()
            }
            else -> {}
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(BlancoAuxiliar)
                .padding(paddingValues)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(bottomStart = 32.dp, bottomEnd = 32.dp))
                    .background(FondoPrincipal)
                    .padding(start = 24.dp, end = 24.dp, top = 48.dp, bottom = 32.dp)
            ) {
                Column {
                    IconButton(
                        onClick = { navController.popBackStack() },
                        modifier = Modifier
                            .size(48.dp)
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
                        text = "Editar perfil",
                        fontSize = 30.sp,
                        fontWeight = FontWeight.Bold,
                        color = FondoBlanco
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Foto de perfil
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .clip(CircleShape)
                    .background(FondoPrincipal)
                    .align(Alignment.CenterHorizontally)
                    .clickable { photoPickerLauncher.launch("image/*") },
                contentAlignment = Alignment.Center
            ) {
                val displayUri = photoUri?.toString() ?: savedPhotoUri
                if (displayUri != null) {
                    AsyncImage(
                        model = displayUri,
                        contentDescription = "Foto de perfil",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    Text(
                        text = user?.nombre?.firstOrNull()?.uppercaseChar()?.toString() ?: "?",
                        color = FondoBlanco,
                        fontWeight = FontWeight.Bold,
                        fontSize = 36.sp
                    )
                }

                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(LetrasNegras.copy(alpha = 0.3f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.CameraAlt,
                        contentDescription = "Cambiar foto",
                        tint = FondoBlanco,
                        modifier = Modifier.size(28.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                OutlinedTextField(
                    value = nombre,
                    onValueChange = { nombre = it },
                    label = { Text("Nombre") },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(32.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = FondoPrincipal,
                        unfocusedBorderColor = FondoPrincipal,
                        focusedLabelColor = FondoPrincipal,
                        unfocusedLabelColor = LetrasNegras50,
                    )
                )

                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text("Email") },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(32.dp),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                    isError = emailFormatError != null,
                    supportingText = {
                        if (emailFormatError != null) {
                            Text(text = emailFormatError, color = MaterialTheme.colorScheme.error)
                        }
                    },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = FondoPrincipal,
                        unfocusedBorderColor = FondoPrincipal,
                        focusedLabelColor = FondoPrincipal,
                        unfocusedLabelColor = LetrasNegras50,
                    )
                )

                OutlinedTextField(
                    value = user?.celular ?: "",
                    onValueChange = {},
                    label = { Text("Celular") },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(32.dp),
                    enabled = false,
                    colors = OutlinedTextFieldDefaults.colors(
                        disabledBorderColor = IconosMenu,
                        disabledLabelColor = LetrasNegras50,
                        disabledTextColor = LetrasNegras50,
                    )
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            Button(
                onClick = {
                    editProfileViewModel.updateProfile(
                        context = context,
                        nombre = nombre,
                        email = email,
                        photoUri = photoUri?.toString()
                    )
                },
                enabled = formValid && state !is EditProfileState.Loading,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 24.dp)
                    .height(56.dp),
                shape = RoundedCornerShape(32.dp),
                colors = ButtonDefaults.buttonColors(containerColor = FondoPrincipal)
            ) {
                if (state is EditProfileState.Loading) {
                    CircularProgressIndicator(
                        color = FondoBlanco,
                        modifier = Modifier.size(24.dp),
                        strokeWidth = 2.dp
                    )
                } else {
                    Text(
                        text = "Actualizar perfil",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = FondoBlanco
                    )
                }
            }
        }
    }
}