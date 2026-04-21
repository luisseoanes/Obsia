package com.upb.obsia.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.upb.obsia.NavRoutes
import com.upb.obsia.R
import com.upb.obsia.ui.theme.*

@Composable
fun OnboardingScreen(navController: NavController) {
        Box(modifier = Modifier.fillMaxSize().background(FondoPrincipal)) {
                Image(
                        painter = painterResource(id = R.drawable.doctor_image),
                        contentDescription = "Doctora",
                        modifier =
                                Modifier.size(600.dp)
                                        .align(Alignment.TopCenter)
                                        .offset(y = 60.dp)
                                        .alpha(0.40f)
                                        .offset(y = 60.dp)
                )

                Column(
                        modifier =
                                Modifier.align(Alignment.Center)
                                        .padding(horizontal = 32.dp)
                                        .offset(y = 140.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                ) {
                        Text(
                                text =
                                        "¡Hola! Puedo ayudarte con tus consultas obstétricas. ¿Qué necesitas?",
                                fontSize = 26.sp,
                                fontWeight = FontWeight.Medium,
                                color = FondoBlanco,
                                textAlign = TextAlign.Center
                        )
                }

                Column(
                        modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 48.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                ) {
                        Button(
                                onClick = {
                                        navController.navigate(NavRoutes.LOGIN) {
                                                popUpTo(NavRoutes.ONBOARDING) { inclusive = true }
                                        }
                                },
                                colors =
                                        ButtonDefaults.buttonColors(
                                                containerColor = FondoBlanco.copy(alpha = 0.3f)
                                        ),
                                shape = RoundedCornerShape(50.dp),
                                modifier =
                                        Modifier.fillMaxWidth()
                                                .padding(horizontal = 48.dp)
                                                .height(52.dp)
                        ) {
                                Text(
                                        text = "Siguiente",
                                        fontSize = 16.sp,
                                        fontWeight = FontWeight.Medium,
                                        color = FondoBlanco
                                )
                        }

                        Spacer(modifier = Modifier.height(2.dp))

                        Text(
                                text = "Arro no sustituye trabajo de un profesional.",
                                fontSize = 14.sp,
                                color = FondoBlanco.copy(alpha = 0.7f),
                                textAlign = TextAlign.Center
                        )
                }
        }
}
