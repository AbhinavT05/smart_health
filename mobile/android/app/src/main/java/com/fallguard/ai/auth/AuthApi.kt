package com.fallguard.ai.auth

import android.content.Context
import com.google.gson.Gson
import com.google.gson.JsonElement
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

data class ApiUser(val id: Int = 0, val name: String = "", val email: String = "", val role: String = "PATIENT")
data class AuthResponse(
    val access_token: String = "",
    val token_type: String = "bearer",
    val expires_in_minutes: Int = 1440,
    val user: ApiUser? = null
)
data class DashboardInfo(
    val name: String,
    val activity: String,
    val risk: String,
    val status: String,
    val sensorConnected: Boolean,
    val falls: Int,
    val activities: Int
)

class ApiException(val statusCode: Int, override val message: String) : Exception(message)

class AuthApi(private val context: Context) {
    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .connectTimeout(12, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .build()
    private val jsonType = "application/json; charset=utf-8".toMediaType()

    suspend fun login(email: String, password: String): AuthResponse = withContext(Dispatchers.IO) {
        val payload = "{\"email\":${gson.toJson(email.trim())},\"password\":${gson.toJson(password)}}"
        execute("/api/auth/login", payload = payload).let(::parseAuth)
    }

    suspend fun register(
        name: String,
        email: String,
        password: String,
        phone: String,
        emergencyName: String,
        emergencyPhone: String,
        emergencyEmail: String
    ): AuthResponse = withContext(Dispatchers.IO) {
        val payload = """{"name":${gson.toJson(name.trim())},"email":${gson.toJson(email.trim())},"password":${gson.toJson(password)},"confirm_password":${gson.toJson(password)},"phone_number":${gson.toJson(phone.trim())},"emergency_contact_name":${gson.toJson(emergencyName.trim())},"emergency_contact_phone":${gson.toJson(emergencyPhone.trim())},"emergency_contact_email":${gson.toJson(emergencyEmail.trim())}}"""
        execute("/api/auth/register", payload = payload).let(::parseAuth)
    }

    suspend fun currentUser(token: String): ApiUser = withContext(Dispatchers.IO) {
        val text = execute("/api/auth/me", token = token)
        gson.fromJson(text, ApiUser::class.java) ?: throw IOException("The backend returned an empty user profile.")
    }

    suspend fun dashboard(token: String): DashboardInfo = withContext(Dispatchers.IO) {
        val root = JsonParser.parseString(execute("/api/dashboard/summary", token = token)).asJsonObject
        val user = root.getAsJsonObject("user")
        DashboardInfo(
            name = user?.get("name")?.asString ?: "FallGuard member",
            activity = root.stringValue("current_activity", "No recent activity"),
            risk = root.stringValue("current_risk", "UNKNOWN"),
            status = root.stringValue("system_status", "READY"),
            sensorConnected = root.booleanValue("sensor_connected"),
            falls = root.intValue("total_falls_count"),
            activities = root.intValue("total_activities_count")
        )
    }

    suspend fun registerDevice(token: String, deviceId: String, displayName: String) = withContext(Dispatchers.IO) {
        val payload = "{\"device_id\":${gson.toJson(deviceId)},\"display_name\":${gson.toJson(displayName)}}"
        execute("/api/auth/devices", method = "POST", payload = payload, token = token)
        Unit
    }

    private fun parseAuth(text: String): AuthResponse {
        val response = gson.fromJson(text, AuthResponse::class.java)
            ?: throw IOException("The backend returned an empty authentication response.")
        if (!response.access_token.matches(Regex("^[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+$"))) {
            throw IOException("The backend response did not contain a valid access token.")
        }
        if (!response.token_type.equals("bearer", ignoreCase = true) || response.user == null) {
            throw IOException("The backend returned an incomplete authentication response.")
        }
        return response
    }

    private fun execute(path: String, method: String = "GET", payload: String? = null, token: String? = null): String {
        val builder = Request.Builder().url(BackendConfig.httpUrl(context, path)).header("Accept", "application/json")
        if (payload != null) builder.header("Content-Type", "application/json")
        if (!token.isNullOrBlank()) builder.header("Authorization", "Bearer $token")
        val body = payload?.toRequestBody(jsonType)
        val request = when (method) {
            "POST" -> builder.post(body ?: ByteArray(0).toRequestBody(null)).build()
            else -> builder.get().build()
        }
        try {
            client.newCall(request).execute().use { response ->
                val responseText = response.body?.string().orEmpty()
                if (!response.isSuccessful) {
                    throw ApiException(response.code, errorMessage(responseText, response.code))
                }
                return responseText
            }
        } catch (error: ApiException) {
            throw error
        } catch (error: IOException) {
            throw IOException("Could not reach ${BackendConfig.getBaseUrl(context)}. ${error.message ?: "Check the backend URL and network."}", error)
        }
    }

    private fun errorMessage(body: String, status: Int): String {
        return try {
            val detail = JsonParser.parseString(body).asJsonObject.get("detail")
            when {
                detail == null || detail.isJsonNull -> "Request failed (HTTP $status)."
                detail.isJsonArray -> detail.asJsonArray.joinToString(" ") { item ->
                    val obj = item.asJsonObject
                    val field = obj.getAsJsonArray("loc")?.lastOrNull()?.asString
                    val message = obj.get("msg")?.asString ?: "Invalid value."
                    if (field.isNullOrBlank()) message else "$field: $message"
                }
                detail.isJsonPrimitive -> detail.asString
                else -> "Request failed (HTTP $status)."
            }
        } catch (_: Exception) {
            "Request failed (HTTP $status)."
        }
    }

    private fun JsonElement.stringValue(key: String, fallback: String) = asJsonObject.get(key)?.takeUnless { it.isJsonNull }?.asString ?: fallback
    private fun JsonElement.booleanValue(key: String) = asJsonObject.get(key)?.asBoolean ?: false
    private fun JsonElement.intValue(key: String) = asJsonObject.get(key)?.asInt ?: 0
}
