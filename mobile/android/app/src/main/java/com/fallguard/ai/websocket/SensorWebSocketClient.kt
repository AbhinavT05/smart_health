package com.fallguard.ai.websocket

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.fallguard.ai.model.Axis3D
import com.fallguard.ai.model.FallConfirmationResponse
import com.fallguard.ai.model.IncomingPendingFallMessage
import com.fallguard.ai.model.RawSensorReading
import com.fallguard.ai.model.SensorDataPacket
import com.google.gson.Gson
import com.google.gson.JsonParser
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

/**
 * Persistent live WebSocket client for streaming 50Hz smartphone telemetry.
 * Automatically recovers with exponential backoff reconnection upon network drops.
 */
class SensorWebSocketClient(
    private var serverUrl: String,
    private var jwtToken: String,
    private var deviceId: String
) {
    private val TAG = "SensorWSClient"
    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile var isConnected = false
        private set

    private var shouldReconnect = true
    private var reconnectDelayMs = 1000L
    private val MAX_RECONNECT_DELAY_MS = 16000L
    @Volatile private var generation = 0
    @Volatile private var authenticationRejected = false
    private var reconnectRunnable: Runnable? = null

    var onConnectionStateChanged: ((Boolean) -> Unit)? = null
    var onPendingFallReceived: ((IncomingPendingFallMessage) -> Unit)? = null
    var onAuthenticationRejected: ((String) -> Unit)? = null

    fun updateConfig(url: String, token: String, devId: String) {
        serverUrl = url
        jwtToken = token
        deviceId = devId
    }

    fun connect() {
        shouldReconnect = true
        authenticationRejected = false
        mainHandler.removeCallbacksAndMessages(null)
        val currentGeneration = ++generation
        webSocket?.cancel()
        webSocket = null
        isConnected = false
        // The backend supports an initial auth frame on /ws/sensor. Send credentials
        // only after the connection opens, never in the URL or logs.
        val request = Request.Builder().url(serverUrl).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                if (currentGeneration != generation) { ws.close(1000, "Superseded connection"); return }
                Log.i(TAG, "Sensor WebSocket transport opened; awaiting authenticated server acknowledgement")
                val authFrame = gson.toJson(mapOf("type" to "auth", "token" to jwtToken, "device_id" to deviceId))
                if (!ws.send(authFrame)) {
                    ws.close(1008, "Authentication frame could not be sent")
                }
            }

            override fun onMessage(ws: WebSocket, text: String) {
                try {
                    val json = JsonParser.parseString(text).asJsonObject
                    when (json.get("type")?.asString) {
                        "auth_success" -> {
                            if (currentGeneration != generation) return
                            isConnected = true
                            reconnectDelayMs = 1000L
                            mainHandler.post { onConnectionStateChanged?.invoke(true) }
                            return
                        }
                        "error" -> {
                            val code = json.get("code")?.asString ?: "AUTH_FAILED"
                            if (code in setOf("AUTH_REQUIRED", "AUTH_FAILED", "USER_NOT_FOUND", "DEVICE_FORBIDDEN", "DEVICE_IN_USE")) {
                                authenticationRejected = true
                                isConnected = false
                                Log.w(TAG, "Sensor WebSocket authorization rejected ($code)")
                                mainHandler.post {
                                    onConnectionStateChanged?.invoke(false)
                                    onAuthenticationRejected?.invoke(code)
                                }
                                ws.close(1008, "Authorization rejected")
                            }
                            return
                        }
                    }
                    val msg = gson.fromJson(text, IncomingPendingFallMessage::class.java)
                    if (msg.type == "FALL_PENDING_CONFIRMATION" && msg.event != null) {
                        Log.w(TAG, "Received potential fall alert from backend! Event ID: ${msg.event.id}")
                        mainHandler.post { onPendingFallReceived?.invoke(msg) }
                    }
                } catch (_: Exception) { /* Ignore non-JSON or unrelated server messages. */ }
            }

            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                Log.w(TAG, "WebSocket closing ($code): $reason")
                isConnected = false
                mainHandler.post { onConnectionStateChanged?.invoke(false) }
                if (code == 1008) authenticationRejected = true
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.w(TAG, "WebSocket closed ($code): $reason")
                isConnected = false
                mainHandler.post { onConnectionStateChanged?.invoke(false) }
                if (currentGeneration == generation) scheduleReconnect(currentGeneration)
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket error: ${t.message}")
                isConnected = false
                mainHandler.post { onConnectionStateChanged?.invoke(false) }
                if (currentGeneration == generation && (response?.code == 401 || response?.code == 403)) {
                    authenticationRejected = true
                    mainHandler.post { onAuthenticationRejected?.invoke("AUTH_FAILED") }
                } else if (currentGeneration == generation) scheduleReconnect(currentGeneration)
            }
        })
    }

    fun disconnect() {
        shouldReconnect = false
        generation++
        mainHandler.removeCallbacksAndMessages(null)
        try {
            webSocket?.close(1000, "Normal closure")
            webSocket = null
        } catch (e: Exception) {
            // Ignore
        }
        isConnected = false
        mainHandler.post { onConnectionStateChanged?.invoke(false) }
    }

    private fun scheduleReconnect(expectedGeneration: Int) {
        if (!shouldReconnect || authenticationRejected) return
        reconnectRunnable?.let(mainHandler::removeCallbacks)
        reconnectRunnable = Runnable {
            if (shouldReconnect && !isConnected && !authenticationRejected && expectedGeneration == generation) {
                Log.i(TAG, "Attempting automatic reconnection (delay ${reconnectDelayMs}ms)...")
                reconnectDelayMs = (reconnectDelayMs * 2).coerceAtMost(MAX_RECONNECT_DELAY_MS)
                connect()
            }
        }
        mainHandler.postDelayed(reconnectRunnable!!, reconnectDelayMs)
    }

    fun sendSensorReading(reading: RawSensorReading, location: Map<String, Double>? = null): Boolean {
        if (!isConnected || authenticationRejected) return false
        val packet = SensorDataPacket(
            deviceId = deviceId,
            timestamp = reading.timestamp,
            unit = "m/s2",
            accelerometer = Axis3D(reading.ax, reading.ay, reading.az),
            gyroscope = Axis3D(reading.gx, reading.gy, reading.gz),
            location = location
        )
        val json = gson.toJson(packet)
        return webSocket?.send(json) ?: false
    }

    fun sendFallConfirmation(eventId: Int, action: String, location: Map<String, Double>? = null): Boolean {
        val payload = FallConfirmationResponse(
            eventId = eventId,
            action = action,
            location = location
        )
        val json = gson.toJson(payload)
        Log.i(TAG, "Sending fall confirmation response: $json")
        return webSocket?.send(json) ?: false
    }
}
