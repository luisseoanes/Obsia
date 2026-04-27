// Ruta: app/src/main/java/com/upb/obsia/di/AppModule.kt

package com.upb.obsia.di

import com.upb.obsia.data.AppDatabase
import com.upb.obsia.data.repository.ChatRepositoryImpl
import com.upb.obsia.data.repository.EngineRepositoryImpl
import com.upb.obsia.domain.EmergencyClinicalRules
import com.upb.obsia.domain.EmergencyDetector
import com.upb.obsia.domain.LlmEngine
import com.upb.obsia.domain.LlmEngineStub
import com.upb.obsia.domain.QueryOrchestrator
import com.upb.obsia.domain.RoutineClinicalRules
import com.upb.obsia.domain.repository.ChatRepository
import com.upb.obsia.domain.repository.EngineRepository
import com.upb.obsia.ui.viewmodel.ChatListViewModel
import com.upb.obsia.ui.viewmodel.ChatViewModel
import org.koin.android.ext.koin.androidContext
import org.koin.core.module.dsl.viewModel
import org.koin.dsl.module

val appModule = module {

    // ─── Base de datos ────────────────────────────────────────────────────────
    single { AppDatabase.getInstance(androidContext()) }
    single { get<AppDatabase>().chatMessageDao() }
    single { get<AppDatabase>().chatSessionDao() }
    single { get<AppDatabase>().userDao() }

    // ─── Repositorios ─────────────────────────────────────────────────────────
    single<EngineRepository> { EngineRepositoryImpl(androidContext()) }
    single<ChatRepository> { ChatRepositoryImpl(get(), get()) }

    // ─── Domain Rules ─────────────────────────────────────────────────────────
    single {
        val json = try {
            androidContext().assets.open("emergency_config.json").bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            android.util.Log.e("ObsIA", "Error cargando emergency_config.json: ${e.message}")
            """{"version":"1.0","categories":[]}"""
        }
        EmergencyDetector(json)
    }
    single {
        val json = try {
            androidContext().assets.open("emergency_clinical_rules.json").bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            android.util.Log.e("ObsIA", "Error cargando emergency_clinical_rules.json: ${e.message}")
            """{"rules":[]}"""
        }
        EmergencyClinicalRules(json)
    }
    single {
        val json = try {
            androidContext().assets.open("routine_clinical_rules.json").bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            android.util.Log.e("ObsIA", "Error cargando routine_clinical_rules.json: ${e.message}")
            """{"rules":[]}"""
        }
        RoutineClinicalRules(json)
    }
    single<LlmEngine> { com.upb.obsia.domain.NativeLlmEngine(get()) }
    single { QueryOrchestrator(get(), get(), get(), get()) }

    // ─── ViewModels ───────────────────────────────────────────────────────────
    viewModel { ChatViewModel(androidContext(), get(), get(), get()) }
    viewModel { ChatListViewModel(androidContext(), get()) }
}
