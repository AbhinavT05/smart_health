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
        isEmulator() -> BuildConfig.DEFAULT_BACKEND_BASE_URL
        !BuildConfig.DEBUG -> BuildConfig.DEFAULT_BACKEND_BASE_URL
        else -> ""
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

    fun httpUrl(context: Context, path: String, queryParameters: Map<String, String> = emptyMap()): HttpUrl {
        val configured = validateBaseUrl(getBaseUrl(context))
            ?: throw IllegalArgumentException("Enter a valid backend base URL. Use http://10.0.2.2:8000 on the Android Emulator, your computer's reachable LAN address on a phone, or a deployed HTTPS URL.")
        val base = configured.toHttpUrlOrNull() ?: error("Invalid backend URL")
        val queryStart = path.indexOf('?')
        val pathPart = (if (queryStart >= 0) path.substring(0, queryStart) else path).trimStart('/')
        val queryPart = if (queryStart >= 0) path.substring(queryStart + 1) else ""
        val builder = base.newBuilder()
        if (pathPart.isNotEmpty()) builder.addPathSegments(pathPart)
        if (queryPart.isNotEmpty()) {
            val queryUrl = "https://fallguard.invalid/?$queryPart".toHttpUrlOrNull()
                ?: throw IllegalArgumentException("Invalid query parameters in backend endpoint.")
            for (index in 0 until queryUrl.querySize) {
                builder.addQueryParameter(queryUrl.queryParameterName(index)!!, queryUrl.queryParameterValue(index))
            }
        }
        queryParameters.forEach { (key, value) -> builder.addQueryParameter(key, value) }
        return builder.build()
    }

    fun sensorWebSocketUrl(context: Context): String {
        val endpoint = httpUrl(context, "/ws/sensor")
        val socketScheme = if (endpoint.isHttps) "wss" else "ws"
        return endpoint.newBuilder().scheme(if (endpoint.isHttps) "https" else "http").build()
            .toString().replaceFirst(Regex("^https?://"), "$socketScheme://")
    }
}
