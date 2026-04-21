package com.upb.obsia

import android.app.Application
import com.upb.obsia.di.appModule
import org.koin.android.ext.koin.androidContext
import org.koin.android.ext.koin.androidLogger
import org.koin.core.context.startKoin
import org.koin.core.logger.Level

class ObsiaApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        startKoin {
            // En release cambiar a Level.ERROR para no pagar el costo de logging
            androidLogger(Level.DEBUG)
            androidContext(this@ObsiaApplication)
            modules(appModule)
        }
    }
}
