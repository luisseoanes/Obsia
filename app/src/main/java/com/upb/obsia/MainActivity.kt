// app/src/main/java/com/upb/obsia/MainActivity.kt

package com.upb.obsia

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.navigation.compose.rememberNavController
import com.upb.obsia.ui.theme.ObsIATheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ObsIATheme {
                val navController = rememberNavController()
                NavGraph(navController = navController)
            }
        }
    }
}
