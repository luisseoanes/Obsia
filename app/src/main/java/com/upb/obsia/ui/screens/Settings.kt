package com.upb.obsia.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ExitToApp
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import androidx.navigation.compose.currentBackStackEntryAsState
import coil.compose.AsyncImage
import com.upb.obsia.NavRoutes
import com.upb.obsia.data.AuthPreferences
import com.upb.obsia.ui.theme.*
import com.upb.obsia.ui.viewmodel.SettingsViewModel

@Composable
fun SettingsScreen(
        navController: NavController,
        settingsViewModel: SettingsViewModel = viewModel()
) {
        val context = LocalContext.current
        val user by settingsViewModel.user.collectAsState()
        val userId = AuthPreferences.getUserId(context)
        val photoUri = AuthPreferences.getPhotoUri(context, userId)
        var showLogoutDialog by remember { mutableStateOf(false) }

        LaunchedEffect(Unit) { settingsViewModel.loadUser(context) }

        Scaffold(
                bottomBar = {
                        val currentRoute =
                                navController.currentBackStackEntryAsState()
                                        .value
                                        ?.destination
                                        ?.route
                                        ?: NavRoutes.SETTINGS
                        Surface(shadowElevation = 16.dp, color = FondoBlanco) {
                                BottomNavBar(
                                        currentRoute = currentRoute,
                                        onNavigateToChats = {
                                                navController.navigate(NavRoutes.CHAT_LIST) {
                                                        launchSingleTop = true
                                                }
                                        },
                                        onNavigateToSettings = {
                                                navController.navigate(NavRoutes.SETTINGS) {
                                                        launchSingleTop = true
                                                }
                                        }
                                )
                        }
                }
        ) { paddingValues ->
                Column(
                        modifier =
                                Modifier.fillMaxSize()
                                        .background(BlancoAuxiliar)
                                        .padding(paddingValues)
                ) {
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
                                                        top = 36.dp,
                                                        bottom = 32.dp
                                                )
                        ) {
                                Text(
                                        text = "Configuración",
                                        fontSize = 30.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = FondoBlanco
                                )
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        Card(
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp),
                                shape = RoundedCornerShape(16.dp),
                                colors = CardDefaults.cardColors(containerColor = FondoBlanco),
                                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                        ) {
                                Row(
                                        modifier = Modifier.fillMaxWidth().padding(16.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                ) {
                                        Box(
                                                modifier =
                                                        Modifier.size(48.dp)
                                                                .clip(CircleShape)
                                                                .background(FondoPrincipal),
                                                contentAlignment = Alignment.Center
                                        ) {
                                                if (photoUri != null) {
                                                        AsyncImage(
                                                                model = photoUri,
                                                                contentDescription =
                                                                        "Foto de perfil",
                                                                contentScale = ContentScale.Crop,
                                                                modifier = Modifier.fillMaxSize()
                                                        )
                                                } else {
                                                        Text(
                                                                text =
                                                                        user?.nombre
                                                                                ?.firstOrNull()
                                                                                ?.uppercaseChar()
                                                                                ?.toString()
                                                                                ?: "?",
                                                                color = FondoBlanco,
                                                                fontWeight = FontWeight.Bold,
                                                                fontSize = 20.sp
                                                        )
                                                }
                                        }

                                        Spacer(modifier = Modifier.width(16.dp))

                                        Column {
                                                Text(
                                                        text = user?.nombre ?: "Cargando...",
                                                        fontWeight = FontWeight.Bold,
                                                        fontSize = 16.sp,
                                                        color = LetrasNegras
                                                )
                                                Text(
                                                        text = user?.celular ?: "",
                                                        fontSize = 14.sp,
                                                        color = LetrasNegras50
                                                )
                                        }
                                }
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        Card(
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp),
                                shape = RoundedCornerShape(16.dp),
                                colors = CardDefaults.cardColors(containerColor = FondoBlanco),
                                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                        ) {
                                Column {
                                        Row(
                                                modifier =
                                                        Modifier.fillMaxWidth()
                                                                .clickable {
                                                                        navController.navigate(
                                                                                NavRoutes
                                                                                        .EDIT_PROFILE
                                                                        )
                                                                }
                                                                .padding(
                                                                        horizontal = 16.dp,
                                                                        vertical = 16.dp
                                                                ),
                                                verticalAlignment = Alignment.CenterVertically
                                        ) {
                                                Icon(
                                                        imageVector = Icons.Filled.Edit,
                                                        contentDescription = "Editar perfil",
                                                        tint = LetrasNegras,
                                                        modifier = Modifier.size(22.dp)
                                                )
                                                Spacer(modifier = Modifier.width(12.dp))
                                                Text(
                                                        text = "Editar perfil",
                                                        fontSize = 15.sp,
                                                        color = LetrasNegras,
                                                        modifier = Modifier.weight(1f)
                                                )
                                                Icon(
                                                        imageVector =
                                                                Icons.AutoMirrored.Filled
                                                                        .KeyboardArrowRight,
                                                        contentDescription = null,
                                                        tint = LetrasNegras50
                                                )
                                        }

                                        HorizontalDivider(
                                                modifier = Modifier.padding(horizontal = 16.dp),
                                                color = IconosMenu
                                        )

                                        Row(
                                                modifier =
                                                        Modifier.fillMaxWidth()
                                                                .clickable {
                                                                        showLogoutDialog = true
                                                                }
                                                                .padding(
                                                                        horizontal = 16.dp,
                                                                        vertical = 16.dp
                                                                ),
                                                verticalAlignment = Alignment.CenterVertically
                                        ) {
                                                Icon(
                                                        imageVector =
                                                                Icons.AutoMirrored.Filled.ExitToApp,
                                                        contentDescription = "Cerrar sesión",
                                                        tint = LetrasNegras,
                                                        modifier = Modifier.size(22.dp)
                                                )
                                                Spacer(modifier = Modifier.width(12.dp))
                                                Text(
                                                        text = "Cerrar sesión",
                                                        fontSize = 15.sp,
                                                        color = LetrasNegras,
                                                        modifier = Modifier.weight(1f)
                                                )
                                                Icon(
                                                        imageVector =
                                                                Icons.AutoMirrored.Filled
                                                                        .KeyboardArrowRight,
                                                        contentDescription = null,
                                                        tint = LetrasNegras50
                                                )
                                        }
                                }
                        }
                }
        }

        if (showLogoutDialog) {
                AlertDialog(
                        onDismissRequest = { showLogoutDialog = false },
                        text = {
                                Text(
                                        text = "¿Seguro desea proceder con el cierre de sesión?",
                                        fontSize = 15.sp,
                                        color = LetrasNegras
                                )
                        },
                        confirmButton = {
                                Button(
                                        onClick = {
                                                AuthPreferences.clearSession(context)
                                                navController.navigate(NavRoutes.LOGIN) {
                                                        popUpTo(0) { inclusive = true }
                                                }
                                        },
                                        colors =
                                                ButtonDefaults.buttonColors(
                                                        containerColor = FondoPrincipal
                                                ),
                                        shape = RoundedCornerShape(50.dp)
                                ) { Text("Sí", color = FondoBlanco) }
                        },
                        dismissButton = {
                                Button(
                                        onClick = { showLogoutDialog = false },
                                        colors =
                                                ButtonDefaults.buttonColors(
                                                        containerColor = FondoPrincipal
                                                ),
                                        shape = RoundedCornerShape(50.dp)
                                ) { Text("No", color = FondoBlanco) }
                        },
                        shape = RoundedCornerShape(16.dp),
                        containerColor = FondoBlanco
                )
        }
}
