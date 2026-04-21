package com.upb.obsia.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val AppColorScheme = lightColorScheme(
    primary = FondoPrincipal,
    onPrimary = FondoBlanco,
    background = FondoBlanco,
    onBackground = LetrasNegras,
    surface = FondoBlanco,
    onSurface = LetrasNegras,
)

@Composable
fun ObsIATheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = AppColorScheme,
        typography = Typography,
        content = content
    )
}