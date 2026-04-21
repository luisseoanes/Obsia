package com.upb.obsia.ui.screens

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddComment
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DriveFileRenameOutline
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.upb.obsia.NavRoutes
import com.upb.obsia.data.ChatSession
import com.upb.obsia.ui.theme.FondoBlanco
import com.upb.obsia.ui.theme.FondoPrincipal
import com.upb.obsia.ui.theme.LetrasNegras
import com.upb.obsia.ui.theme.LetrasNegras50
import com.upb.obsia.ui.viewmodel.ChatListViewModel
import org.koin.compose.viewmodel.koinViewModel

private val IconosMenuOscuro = Color(0xFF8A8A8A)

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun ChatList(
        onNavigateToChat: (sessionId: Int) -> Unit,
        onNavigateToSettings: () -> Unit,
        viewModel: ChatListViewModel = koinViewModel() // koinViewModel() reemplaza viewModel()
) {
        val sessions by viewModel.sessions.collectAsState()
        val searchQuery by viewModel.searchQuery.collectAsState()
        val newSessionId by viewModel.newSessionId.collectAsState()
        val selectedSession by viewModel.selectedSession.collectAsState()
        val errorMessage by viewModel.errorMessage.collectAsState()
        val lastMessagePreviews by viewModel.lastMessagePreviews.collectAsState()
        val snackbarHostState = remember { SnackbarHostState() }

        var sessionForMenu by remember { mutableStateOf<ChatSession?>(null) }
        var sessionToDelete by remember { mutableStateOf<ChatSession?>(null) }

        // Context ya no se pasa - el ViewModel lo obtiene internamente
        LaunchedEffect(Unit) { viewModel.loadSessions() }

        LaunchedEffect(newSessionId) {
                newSessionId?.let { id ->
                        viewModel.onNavigationHandled()
                        onNavigateToChat(id)
                }
        }

        LaunchedEffect(errorMessage) {
                errorMessage?.let {
                        snackbarHostState.showSnackbar(it)
                        viewModel.clearError()
                }
        }

        selectedSession?.let { session ->
                RenameDialog(
                        currentTitle = session.title,
                        onConfirm = { newTitle ->
                                viewModel.renameSession(session, newTitle)
                        }, // sin context
                        onDismiss = { viewModel.dismissRenameDialog() }
                )
        }

        sessionToDelete?.let { session ->
                DeleteConfirmDialog(
                        sessionTitle = session.title,
                        onConfirm = {
                                viewModel.deleteSession(session) // sin context
                                sessionToDelete = null
                        },
                        onDismiss = { sessionToDelete = null }
                )
        }

        Scaffold(
                snackbarHost = { SnackbarHost(snackbarHostState) },
                containerColor = FondoBlanco,
                floatingActionButton = {
                        FloatingActionButton(
                                onClick = { viewModel.createNewSession() },
                                containerColor = FondoPrincipal,
                                contentColor = FondoBlanco,
                                shape = CircleShape,
                                modifier = Modifier.size(60.dp)
                        ) {
                                Icon(
                                        imageVector = Icons.Filled.AddComment,
                                        contentDescription = "Nueva conversación",
                                        modifier = Modifier.size(26.dp)
                                )
                        }
                },
                bottomBar = {
                        Surface(shadowElevation = 16.dp, color = FondoBlanco) {
                                BottomNavBar(
                                        currentRoute = NavRoutes.CHAT_LIST,
                                        onNavigateToChats = {},
                                        onNavigateToSettings = onNavigateToSettings
                                )
                        }
                }
        ) { innerPadding ->
                Column(modifier = Modifier.fillMaxSize().padding(innerPadding)) {
                        TopSearchBar(
                                searchQuery = searchQuery,
                                onSearchQueryChange = { viewModel.onSearchQueryChange(it) }
                        )
                        if (sessions.isEmpty()) {
                                EmptyState()
                        } else {
                                SessionList(
                                        sessions = sessions,
                                        lastMessagePreviews = lastMessagePreviews,
                                        onSessionClick = onNavigateToChat,
                                        onSessionLongPress = { session ->
                                                sessionForMenu = session
                                        },
                                        sessionForMenu = sessionForMenu,
                                        onDismissMenu = { sessionForMenu = null },
                                        onRename = { session ->
                                                viewModel.onSessionLongPress(session)
                                        },
                                        onDelete = { session -> sessionToDelete = session }
                                )
                        }
                }
        }
}

// Header

@Composable
private fun TopSearchBar(searchQuery: String, onSearchQueryChange: (String) -> Unit) {
        Box(
                modifier =
                        Modifier.fillMaxWidth()
                                .background(FondoPrincipal)
                                .windowInsetsPadding(WindowInsets.statusBars)
                                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
                Column {
                        Row(
                                modifier =
                                        Modifier.fillMaxWidth()
                                                .clip(RoundedCornerShape(50.dp))
                                                .background(FondoBlanco)
                                                .padding(horizontal = 4.dp, vertical = 2.dp),
                                verticalAlignment = Alignment.CenterVertically
                        ) {
                                OutlinedTextField(
                                        value = searchQuery,
                                        onValueChange = onSearchQueryChange,
                                        placeholder = {
                                                Text(
                                                        text = "Buscar un paciente por nombre",
                                                        color = LetrasNegras50,
                                                        fontSize = 14.sp
                                                )
                                        },
                                        singleLine = true,
                                        keyboardOptions =
                                                KeyboardOptions(imeAction = ImeAction.Search),
                                        colors =
                                                OutlinedTextFieldDefaults.colors(
                                                        focusedBorderColor = Color.Transparent,
                                                        unfocusedBorderColor = Color.Transparent,
                                                        focusedContainerColor = Color.Transparent,
                                                        unfocusedContainerColor = Color.Transparent,
                                                        cursorColor = FondoPrincipal
                                                ),
                                        modifier = Modifier.weight(1f)
                                )
                                Icon(
                                        imageVector = Icons.Filled.Search,
                                        contentDescription = null,
                                        tint = LetrasNegras,
                                        modifier = Modifier.size(36.dp).padding(end = 8.dp)
                                )
                        }
                }
        }
}

// Lista

@Composable
private fun SessionList(
        sessions: List<ChatSession>,
        lastMessagePreviews: Map<Int, String>,
        onSessionClick: (Int) -> Unit,
        onSessionLongPress: (ChatSession) -> Unit,
        sessionForMenu: ChatSession?,
        onDismissMenu: () -> Unit,
        onRename: (ChatSession) -> Unit,
        onDelete: (ChatSession) -> Unit
) {
        LazyColumn(contentPadding = PaddingValues(vertical = 8.dp)) {
                items(items = sessions, key = { it.id }) { session ->
                        SessionItem(
                                session = session,
                                lastMessagePreview = lastMessagePreviews[session.id]
                                                ?: "¡Arro está aquí para ayudarte!",
                                onClick = { onSessionClick(session.id) },
                                onLongPress = { onSessionLongPress(session) },
                                showMenu = sessionForMenu?.id == session.id,
                                onDismissMenu = onDismissMenu,
                                onRename = { onRename(session) },
                                onDelete = { onDelete(session) }
                        )
                }
        }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun SessionItem(
        session: ChatSession,
        lastMessagePreview: String,
        onClick: () -> Unit,
        onLongPress: () -> Unit,
        showMenu: Boolean,
        onDismissMenu: () -> Unit,
        onRename: () -> Unit,
        onDelete: () -> Unit
) {
        Box {
                Row(
                        modifier =
                                Modifier.fillMaxWidth()
                                        .background(
                                                if (showMenu) Color(0xFFF0F0F0) else FondoBlanco
                                        )
                                        .combinedClickable(
                                                onClick = onClick,
                                                onLongClick = onLongPress,
                                                onClickLabel = "Abrir conversación",
                                                onLongClickLabel = "Opciones de conversación"
                                        )
                                        .padding(horizontal = 20.dp, vertical = 14.dp),
                        verticalAlignment = Alignment.CenterVertically
                ) {
                        Column(modifier = Modifier.weight(1f)) {
                                Text(
                                        text = session.title,
                                        color = LetrasNegras,
                                        fontSize = 15.sp,
                                        fontWeight = FontWeight.Medium,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                )
                                Spacer(modifier = Modifier.height(2.dp))
                                Text(
                                        text = lastMessagePreview,
                                        color = LetrasNegras50,
                                        fontSize = 13.sp,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                )
                        }
                }

                DropdownMenu(expanded = showMenu, onDismissRequest = onDismissMenu) {
                        DropdownMenuItem(
                                text = { Text("Renombrar") },
                                onClick = {
                                        onDismissMenu()
                                        onRename()
                                },
                                leadingIcon = {
                                        Icon(
                                                imageVector = Icons.Filled.DriveFileRenameOutline,
                                                contentDescription = null
                                        )
                                }
                        )
                        DropdownMenuItem(
                                text = { Text("Eliminar", color = Color.Red) },
                                onClick = {
                                        onDismissMenu()
                                        onDelete()
                                },
                                leadingIcon = {
                                        Icon(
                                                imageVector = Icons.Filled.Delete,
                                                contentDescription = null,
                                                tint = Color.Red
                                        )
                                }
                        )
                }
        }
}

// Estado vacío

@Composable
private fun EmptyState() {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                        text = "No hay conversaciones disponibles.\n¡Vamos a crear una!",
                        color = LetrasNegras50,
                        fontSize = 15.sp,
                        lineHeight = 22.sp,
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
        }
}

// Diálogo renombrar

@Composable
private fun RenameDialog(currentTitle: String, onConfirm: (String) -> Unit, onDismiss: () -> Unit) {
        var text by remember(currentTitle) { mutableStateOf(currentTitle) }

        AlertDialog(
                onDismissRequest = onDismiss,
                containerColor = FondoBlanco,
                title = {
                        Text(
                                text = "Renombrar conversación",
                                fontWeight = FontWeight.SemiBold,
                                color = LetrasNegras
                        )
                },
                text = {
                        OutlinedTextField(
                                value = text,
                                onValueChange = { text = it },
                                singleLine = true,
                                label = { Text("Nombre", color = LetrasNegras50) },
                                colors =
                                        OutlinedTextFieldDefaults.colors(
                                                focusedBorderColor = FondoPrincipal,
                                                unfocusedBorderColor = LetrasNegras50,
                                                cursorColor = FondoPrincipal,
                                                focusedLabelColor = FondoPrincipal
                                        )
                        )
                },
                confirmButton = {
                        TextButton(
                                onClick = { onConfirm(text) },
                                colors =
                                        ButtonDefaults.textButtonColors(
                                                contentColor = FondoPrincipal
                                        )
                        ) { Text("Guardar", fontWeight = FontWeight.SemiBold) }
                },
                dismissButton = {
                        TextButton(
                                onClick = onDismiss,
                                colors =
                                        ButtonDefaults.textButtonColors(
                                                contentColor = LetrasNegras50
                                        )
                        ) { Text("Cancelar") }
                }
        )
}

// Diálogo confirmar eliminación

@Composable
private fun DeleteConfirmDialog(
        sessionTitle: String,
        onConfirm: () -> Unit,
        onDismiss: () -> Unit
) {
        AlertDialog(
                onDismissRequest = onDismiss,
                containerColor = FondoBlanco,
                title = {
                        Text(
                                text = "Eliminar conversación",
                                fontWeight = FontWeight.SemiBold,
                                color = LetrasNegras
                        )
                },
                text = {
                        Text(
                                text =
                                        "¿Estás seguro de que deseas eliminar \"$sessionTitle\"? Esta acción no se puede deshacer.",
                                color = LetrasNegras50,
                                fontSize = 14.sp
                        )
                },
                confirmButton = {
                        TextButton(
                                onClick = onConfirm,
                                colors = ButtonDefaults.textButtonColors(contentColor = Color.Red)
                        ) { Text("Eliminar", fontWeight = FontWeight.SemiBold) }
                },
                dismissButton = {
                        TextButton(
                                onClick = onDismiss,
                                colors =
                                        ButtonDefaults.textButtonColors(
                                                contentColor = LetrasNegras50
                                        )
                        ) { Text("Cancelar") }
                }
        )
}
