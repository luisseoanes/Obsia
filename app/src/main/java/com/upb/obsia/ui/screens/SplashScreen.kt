package com.upb.obsia.ui.screens
import com.upb.obsia.ui.theme.MensajesUsuario
import androidx.compose.ui.draw.alpha

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.upb.obsia.NavRoutes
import com.upb.obsia.R
import com.upb.obsia.ui.theme.FondoPrincipal
import kotlinx.coroutines.delay
import androidx.compose.material3.Text
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

@Composable
fun SplashScreen(navController: NavController) {
    LaunchedEffect(Unit) {
        delay(2000)
        navController.navigate(NavRoutes.ONBOARDING) {
            popUpTo(NavRoutes.SPLASH) { inclusive = true }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(FondoPrincipal)
    ) {
        Image(
            painter = painterResource(id = R.drawable.arro_logo),
            contentDescription = "Arro logo",
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 80.dp)
                .height(58.dp)
                .width(160.dp),
            colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(MensajesUsuario)
        )

        Column(
            modifier = Modifier
                .align(Alignment.Center)
                .offset(y = 50.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = R.drawable.obstetrician_image),
                contentDescription = "Ícono obstetricia",
                modifier = Modifier
                    .size(260.dp)
                    .alpha(0.35f),
                colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(MensajesUsuario)
            )

            Spacer(modifier = Modifier.height(96.dp))

            Text(
                text = "Arro",
                fontSize = 40.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Tu asistente local",
                fontSize = 24.sp,
                color = Color.White.copy(alpha = 0.85f)
            )
        }
    }
}