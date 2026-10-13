package com.fallguard.ai.auth

import android.content.Context
import android.os.Build
import com.fallguard.ai.BuildConfig
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

object BackendConfig {
    private const val PREFS = "fallguard_settings"
    private const val BASE_URL = "backend_base_url"

    private fun isEmulator(): Boolean =
        Build.FINGERPRINT.startsWith("generic") || Build.FINGERPRINT.lowercase().contains("emulator") ||
            Build.MODEL.contains("Emulator", ignoreCase = true) || Build.HARDWARE.contains("goldfish") ||
            Build.HARDWARE.contains("ranchu") || Build.PRODUCT.contains("sdk", ignoreCase = true)

    fun defaultBaseUrl(): String = when {
        !BuildConfig.DEBUG -> BuildConfig.DEFAULT_BACKEND_BASE_URL
        isEmulator() -> BuildConfig.DEFAULT_BACKEND_BASE_URL
        BuildConfig.REAL_DEVICE_DEFAULT_BACKEND_BASE_URL.isNotBlank() -> BuildConfig.REAL_DEVICE_DEFAULT_BACKEND_BASE_URL
        else -> BuildConfig.DEFAULT_BACKEND_BASE_URL
    }

    fun getBaseUrl(context: Context): String = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        .getString(BASE_URL, null) ?: defaultBaseUrl()

    fun validateBaseUrl(raw: String): String? {
        val candidate = raw.trim().trimEnd('/')
        val parsed = candidate.toHttpUrlOrNull() ?: return null
        if (parsed.scheme != "http" && parsed.scheme != "https") return null
        if (parsed.host.equals("localhost", ignoreCase = true) || parsed.host == "127.0.0.1" || parsed.host == "::1") return null
        if (!BuildConfig.DEBUG && parsed.scheme != "https") return null
        if (parsed.username.isNotEmpty() || parsed.password.isNotEmpty() || parsed.query != null || parsed.fragment != null) return null
        if (parsed.encodedPath.isNotEmpty() && parsed.encodedPath != "/") return null
        return candidate
    }

    fun saveBaseUrl(context: Context, raw: String): String? {
        val normalized = validateBaseUrl(raw) ?: return null
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(BASE_URL, normalized).apply()
        return normalized
    }

    fun httpUrl(context: Context, path: String): HttpUrl {
        val base = (validateBaseUrl(getBaseUrl(context)) ?: BuildConfig.DEFAULT_BACKEND_BASE_URL).toHttpUrlOrNull()
            ?: error("Invalid backend URL")
        return base.newBuilder().addPathSegments(path.removePrefix("/")).build()
    }

    fun sensorWebSocketUrl(context: Context): String {
        val base = getBaseUrl(context)
        val wsBase = when {
            base.startsWith("https://", true) -> "wss://${base.substringAfter("://")}" 
            base.startsWith("http://", true) -> "ws://${base.substringAfter("://")}" 
            else -> error("Backend URL must use http or https")
        }
        return "$wsBase/ws/sensor"
    }
}
