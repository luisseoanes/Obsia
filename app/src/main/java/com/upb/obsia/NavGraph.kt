package com.upb.obsia

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.upb.obsia.ui.screens.ChatList
import com.upb.obsia.ui.screens.ChatScreen
import com.upb.obsia.ui.screens.EditProfileScreen
import com.upb.obsia.ui.screens.LoginScreen
import com.upb.obsia.ui.screens.OnboardingScreen
import com.upb.obsia.ui.screens.RegisterScreen
import com.upb.obsia.ui.screens.SettingsScreen
import com.upb.obsia.ui.screens.SplashScreen

@Composable
fun NavGraph(navController: NavHostController) {
    NavHost(navController = navController, startDestination = NavRoutes.SPLASH) {
        composable(NavRoutes.SPLASH) { SplashScreen(navController) }
        composable(NavRoutes.ONBOARDING) { OnboardingScreen(navController) }
        composable(NavRoutes.REGISTER) { RegisterScreen(navController) }
        composable(NavRoutes.LOGIN) { LoginScreen(navController) }
        composable(NavRoutes.EDIT_PROFILE) { EditProfileScreen(navController) }
        composable(NavRoutes.SETTINGS) { SettingsScreen(navController) }

        composable(NavRoutes.CHAT_LIST) {
            ChatList(
                    onNavigateToChat = { sessionId ->
                        navController.navigate("${NavRoutes.CHAT_SCREEN}/$sessionId")
                    },
                    onNavigateToSettings = { navController.navigate(NavRoutes.SETTINGS) }
            )
        }

        // Ruta simplificada — solo sessionId
        // userId lo resuelve ChatViewModel desde AuthPreferences
        // sessionName lo resuelve ChatViewModel desde la DB
        composable(
                route = "${NavRoutes.CHAT_SCREEN}/{sessionId}",
                arguments = listOf(navArgument("sessionId") { type = NavType.IntType })
        ) { backStackEntry ->
            val sessionId = backStackEntry.arguments?.getInt("sessionId") ?: 0
            ChatScreen(sessionId = sessionId, onNavigateBack = { navController.popBackStack() })
        }
    }
}
