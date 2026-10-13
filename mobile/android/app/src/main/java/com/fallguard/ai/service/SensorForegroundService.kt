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
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.fallguard.ai.auth.BackendConfig
import com.fallguard.ai.auth.SecureSessionStore
import com.fallguard.ai.model.FallResolution
import com.fallguard.ai.sensor.SensorCollector
import com.fallguard.ai.ui.FallAlertActivity
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
    private val raisedFallAlerts = mutableSetOf<Int>()
    private val fallAlertListeners = mutableMapOf<Int, (FallResolution) -> Unit>()
    private val resolvedFallAlerts = linkedMapOf<Int, FallResolution>()

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

        // Update the existing socket exactly once, or create and connect a new one.
        // updateConfig() closes a superseded socket and starts the replacement itself.
        val existingClient = webSocketClient
        if (existingClient == null) {
            webSocketClient = SensorWebSocketClient(serverUrl, jwtToken, deviceId).apply {
                onPendingFallReceived = { incoming ->
                    incoming.event?.let { evt ->
                        val eventId = evt.eventId.takeIf { it > 0 } ?: evt.id
                        launchFallConfirmationScreen(eventId, evt.timeoutSeconds, evt.confidence)
                    }
                }
                onFallResolved = ::handleFallResolved
                onAuthenticationRejected = { code -> this@SensorForegroundService.onAuthenticationRejected?.invoke(code) }
            }
            webSocketClient?.connect()
        } else {
            existingClient.onAuthenticationRejected = { code -> this@SensorForegroundService.onAuthenticationRejected?.invoke(code) }
            existingClient.onFallResolved = ::handleFallResolved
            existingClient.updateConfig(serverUrl, jwtToken, deviceId)
        }

        // Hook up sensor collector to stream live samples
        sensorCollector.onReadingListener = { reading ->
            webSocketClient?.sendSensorReading(reading)
        }
        sensorCollector.start()

        Log.i(TAG, "SensorForegroundService started with deviceId=$deviceId")
        return START_STICKY
    }

    private fun launchFallConfirmationScreen(eventId: Int, timeoutSeconds: Float, confidence: Float) {
        if (eventId <= 0 || !raisedFallAlerts.add(eventId)) {
            Log.i(TAG, "Ignoring duplicate or invalid pending fall alert eventId=$eventId")
            return
        }
        val intent = Intent(this, FallAlertActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra(FallAlertActivity.EXTRA_EVENT_ID, eventId)
            putExtra(FallAlertActivity.EXTRA_DEADLINE, SystemClock.elapsedRealtime() + (timeoutSeconds.coerceAtLeast(1f) * 1000).toLong())
            putExtra("CONFIDENCE", confidence)
        }
        val pending = PendingIntent.getActivity(
            this, eventId, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        if (!NotificationManagerCompat.from(this).areNotificationsEnabled()) {
            Log.w(TAG, "Fall alert notification suppressed because notification permission is unavailable")
            return
        }
        if (Build.VERSION.SDK_INT >= 34 && !getSystemService(NotificationManager::class.java).canUseFullScreenIntent()) {
            Log.w(TAG, "Full-screen notification access is disabled by Android; showing the permitted notification fallback")
        }
        val notification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("FALL DETECTED!")
            .setContentText("Are you okay? Confirm before the ${timeoutSeconds.toInt()} second timer expires.")
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setOngoing(true)
            .setAutoCancel(false)
            .setVibrate(longArrayOf(0, 500, 200, 500, 200, 800))
            .setContentIntent(pending)
            .setFullScreenIntent(pending, true)
            .build()
        try {
            NotificationManagerCompat.from(this).notify(eventId, notification)
        } catch (error: SecurityException) {
            raisedFallAlerts.remove(eventId)
            Log.e(TAG, "Android denied fall-alert notification permission", error)
        }
    }

    fun dismissFallNotification(eventId: Int) {
        (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(eventId)
    }

    private fun handleFallResolved(result: FallResolution) {
        dismissFallNotification(result.eventId)
        raisedFallAlerts.remove(result.eventId)
        resolvedFallAlerts[result.eventId] = result
        if (resolvedFallAlerts.size > 64) resolvedFallAlerts.remove(resolvedFallAlerts.keys.first())
        fallAlertListeners.remove(result.eventId)?.invoke(result)
    }

    fun setFallAlertListener(eventId: Int, listener: (FallResolution) -> Unit) {
        val alreadyResolved = resolvedFallAlerts[eventId]
        if (alreadyResolved != null) listener(alreadyResolved) else fallAlertListeners[eventId] = listener
    }

    fun clearFallAlertListener(eventId: Int) {
        fallAlertListeners.remove(eventId)
    }

    fun submitFallConfirmation(eventId: Int, action: String, location: Map<String, Double>?): Boolean {
        return webSocketClient?.sendFallConfirmation(eventId, action, location) ?: false
    }

    fun onNetworkAvailable() {
        webSocketClient?.reconnectForNetworkChange()
    }

    override fun onDestroy() {
        super.onDestroy()
        onAuthenticationRejected = null
        fallAlertListeners.clear()
        resolvedFallAlerts.clear()
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
