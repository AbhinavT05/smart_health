package com.fallguard.ai.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.fallguard.ai.auth.BackendConfig
import com.fallguard.ai.auth.SecureSessionStore
import com.fallguard.ai.sensor.SensorCollector
import com.fallguard.ai.ui.FallConfirmationActivity
import com.fallguard.ai.ui.MainActivity
import com.fallguard.ai.websocket.SensorWebSocketClient

/**
 * Android Foreground Service to maintain uninterrupted 50Hz sensor collection
 * and live WebSocket telemetry streaming even when the app is in the background or screen is off.
 */
class SensorForegroundService : Service() {

    private val TAG = "SensorForegroundService"
    private val CHANNEL_ID = "fallguard_sensor_channel"
    private val NOTIFICATION_ID = 1001
    private val ALERT_CHANNEL_ID = "fallguard_fall_alerts"

    private val binder = LocalBinder()
    lateinit var sensorCollector: SensorCollector
    private var cpuWakeLock: PowerManager.WakeLock? = null
    var webSocketClient: SensorWebSocketClient? = null
    var onAuthenticationRejected: ((String) -> Unit)? = null

    inner class LocalBinder : Binder() {
        fun getService(): SensorForegroundService = this@SensorForegroundService
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        sensorCollector = SensorCollector(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val storedSession = SecureSessionStore.read(this)
        val jwtToken = intent?.getStringExtra("JWT_TOKEN")?.takeIf { it.isNotBlank() } ?: storedSession?.accessToken.orEmpty()
        if (jwtToken.isBlank()) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        val serverUrl = intent?.getStringExtra("SERVER_URL") ?: BackendConfig.sensorWebSocketUrl(this)
        val deviceId = intent?.getStringExtra("DEVICE_ID")
            ?: getSharedPreferences("fallguard_settings", MODE_PRIVATE).getString("sensor_device_id", "android_phone")
            ?: "android_phone"

        startForeground(NOTIFICATION_ID, buildNotification("Active 50Hz Sensor Stream"))
        if (cpuWakeLock?.isHeld != true) {
            val power = getSystemService(Context.POWER_SERVICE) as PowerManager
            cpuWakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "FallGuardAI:SensorStream").apply {
                setReferenceCounted(false)
                acquire()
            }
        }

        // Initialize WebSocket Client
        if (webSocketClient == null) {
            webSocketClient = SensorWebSocketClient(serverUrl, jwtToken, deviceId).apply {
                onPendingFallReceived = { incoming ->
                    incoming.event?.let { evt ->
                        launchFallConfirmationScreen(evt.eventId, evt.timeoutSeconds, evt.confidence)
                    }
                }
                onFallResolved = { eventId -> dismissFallNotification(eventId) }
                onAuthenticationRejected = { code -> this@SensorForegroundService.onAuthenticationRejected?.invoke(code) }
            }
        } else {
            webSocketClient?.updateConfig(serverUrl, jwtToken, deviceId)
            webSocketClient?.onAuthenticationRejected = { code -> this@SensorForegroundService.onAuthenticationRejected?.invoke(code) }
            webSocketClient?.onFallResolved = { eventId -> dismissFallNotification(eventId) }
        }

        webSocketClient?.connect()

        // Hook up sensor collector to stream live samples
        sensorCollector.onReadingListener = { reading ->
            webSocketClient?.sendSensorReading(reading)
        }
        sensorCollector.start()

        Log.i(TAG, "SensorForegroundService started with deviceId=$deviceId")
        return START_STICKY
    }

    private fun launchFallConfirmationScreen(eventId: Int, timeoutSeconds: Float, confidence: Float) {
        val intent = Intent(this, FallConfirmationActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra("EVENT_ID", eventId)
            putExtra("TIMEOUT_SECONDS", timeoutSeconds)
            putExtra("CONFIDENCE", confidence)
        }
        val pending = PendingIntent.getActivity(
            this, eventId, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("Possible fall detected")
            .setContentText("Tap to confirm you are okay. Automatic escalation in ${timeoutSeconds.toInt()} seconds.")
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setAutoCancel(true)
            .setVibrate(longArrayOf(0, 500, 200, 500, 200, 800))
            .setContentIntent(pending)
            .setFullScreenIntent(pending, true)
            .build()
        (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).notify(eventId, notification)
    }

    fun dismissFallNotification(eventId: Int) {
        (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(eventId)
    }

    fun submitFallConfirmation(eventId: Int, action: String, location: Map<String, Double>?) {
        webSocketClient?.sendFallConfirmation(eventId, action, location)
    }

    override fun onDestroy() {
        super.onDestroy()
        onAuthenticationRejected = null
        sensorCollector.stop()
        webSocketClient?.disconnect()
        cpuWakeLock?.let { if (it.isHeld) it.release() }
        cpuWakeLock = null
        Log.i(TAG, "SensorForegroundService stopped.")
    }

    override fun onBind(intent: Intent?): IBinder = binder

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "FallGuard Sensor Monitor",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Continuous 50Hz Fall Detection Sensor Ingestion Service"
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
            val alertChannel = NotificationChannel(
                ALERT_CHANNEL_ID,
                "Fall confirmation alerts",
                NotificationManager.IMPORTANCE_HIGH,
            ).apply {
                description = "High-priority confirmation prompt for model-detected falls"
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 500, 200, 500, 200, 800)
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            }
            manager.createNotificationChannel(alertChannel)
        }
    }

    private fun buildNotification(statusText: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("FallGuard AI Active")
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
