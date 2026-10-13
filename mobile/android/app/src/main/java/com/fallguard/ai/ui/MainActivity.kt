package com.fallguard.ai.ui

import android.Manifest
import android.app.AlertDialog
import android.app.NotificationManager
import android.net.ConnectivityManager
import android.net.Network
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.provider.Settings
import android.text.InputType
import android.widget.EditText
import android.widget.LinearLayout
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.fallguard.ai.R
import com.fallguard.ai.auth.ApiException
import com.fallguard.ai.auth.AuthApi
import com.fallguard.ai.auth.AuthSession
import com.fallguard.ai.auth.ApiUser
import com.fallguard.ai.auth.BackendConfig
import com.fallguard.ai.auth.SecureSessionStore
import com.fallguard.ai.service.SensorForegroundService
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.launch

/** Authenticated smartphone sensor dashboard and monitor controls. */
class MainActivity : AppCompatActivity() {
    private val permissionsRequestCode = 200
    private lateinit var tvStatus: TextView
    private lateinit var tvBufferInfo: TextView
    private lateinit var tvSensorPreview: TextView
    private lateinit var tvDashboardSummary: TextView
    private lateinit var tvSignedInUser: TextView
    private lateinit var etBackendUrl: TextInputEditText
    private lateinit var etDeviceId: TextInputEditText
    private lateinit var btnToggleMonitoring: MaterialButton
    private lateinit var btnLogout: MaterialButton
    private lateinit var btnUpdateContact: MaterialButton
    private lateinit var btnSaveBackendUrl: MaterialButton
    private lateinit var btnTestBackendUrl: MaterialButton
    private lateinit var tvBackendConnectionResult: TextView
    private lateinit var session: AuthSession
    private var signedInProfile: ApiUser? = null
    private val authApi by lazy { AuthApi(this) }

    private var service: SensorForegroundService? = null
    private var isMonitoringActive = false
    private var bound = false
    private val mainHandler = Handler(Looper.getMainLooper())
    private val networkManager by lazy { getSystemService(ConnectivityManager::class.java) }
    private var lastKnownNetwork: Network? = null
    private var networkCallbackRegistered = false
    private val networkDashboardRefresh = Runnable {
        if (!isFinishing && ::session.isInitialized && session.isValid()) loadDashboardOnce()
    }
    private val backendNetworkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            if (network == lastKnownNetwork) return
            lastKnownNetwork = network
            mainHandler.post {
                if (isMonitoringActive) service?.onNetworkAvailable()
                mainHandler.removeCallbacks(networkDashboardRefresh)
                mainHandler.postDelayed(networkDashboardRefresh, 700L)
            }
        }

        override fun onLost(network: Network) {
            if (lastKnownNetwork == network) {
                lastKnownNetwork = null
                mainHandler.post {
                    if (networkManager.activeNetwork == null) {
                        tvStatus.text = "Network unavailable — waiting to reconnect"
                        tvStatus.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.warning))
                    }
                }
            }
        }
    }
    private val statsRunnable = object : Runnable {
        override fun run() {
            service?.let { s ->
                val sc = s.sensorCollector
                tvSensorPreview.text = String.format(
                    "Acc:  (%.2f, %.2f, %.2f) m/s²\nGyro: (%.2f, %.2f, %.2f) rad/s",
                    sc.latestAx, sc.latestAy, sc.latestAz, sc.latestGx, sc.latestGy, sc.latestGz
                )
                tvBufferInfo.text = String.format(
                    "Rolling Buffer: %d / 250 (~%.1fs)\nRate: acc %.1f Hz, gyro %.1f Hz, output %.1f Hz | skipped grid points: %d",
                    sc.rollingBuffer.size, sc.rollingBuffer.size / 50.0,
                    sc.measuredAccelHz, sc.measuredGyroHz, sc.emittedHz, sc.droppedGridSamples,
                )
            }
            mainHandler.postDelayed(this, 200)
        }
    }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as? SensorForegroundService.LocalBinder)?.getService()
            service?.onAuthenticationRejected = { code ->
                if (code == "AUTH_FAILED" || code == "AUTH_REQUIRED" || code == "USER_NOT_FOUND") {
                    runOnUiThread { expireSession("The server rejected your session. Please sign in again.") }
                } else {
                    runOnUiThread {
                        stopMonitoring()
                        tvDashboardSummary.text = "Monitoring authorization rejected ($code). Check device ownership and try again."
                    }
                }
            }
            service?.webSocketClient?.onConnectionStateChanged = { connected ->
                tvStatus.text = if (connected) getString(R.string.status_connected) else getString(R.string.status_disconnected)
                tvStatus.setTextColor(ContextCompat.getColor(this@MainActivity, if (connected) R.color.success else R.color.danger))
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) { service = null }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        session = SecureSessionStore.read(this) ?: run {
            goToLogin("Please sign in to access your dashboard.")
            return
        }
        setContentView(R.layout.activity_main)
        initViews()
        requestAppPermissions()
        loadDashboardOnce()
        mainHandler.post(statsRunnable)
        showFirstRunPlacementInstructions()
    }

    private fun initViews() {
        tvStatus = findViewById(R.id.tv_connection_status)
        tvBufferInfo = findViewById(R.id.tv_buffer_info)
        tvSensorPreview = findViewById(R.id.tv_sensor_preview)
        tvDashboardSummary = findViewById(R.id.tv_dashboard_summary)
        tvSignedInUser = findViewById(R.id.tv_signed_in_user)
        etBackendUrl = findViewById(R.id.et_server_url)
        etDeviceId = findViewById(R.id.et_device_id)
        btnToggleMonitoring = findViewById(R.id.btn_toggle_monitoring)
        btnLogout = findViewById(R.id.btn_logout)
        btnUpdateContact = findViewById(R.id.btn_update_emergency_contact)
        btnSaveBackendUrl = findViewById(R.id.btn_save_backend_url)
        btnTestBackendUrl = findViewById(R.id.btn_test_backend_url)
        tvBackendConnectionResult = findViewById(R.id.tv_backend_connection_result)

        etBackendUrl.setText(BackendConfig.getBaseUrl(this))
        val defaultDevice = "android_${Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)?.take(32) ?: "phone"}"
        etDeviceId.setText(getSharedPreferences("fallguard_settings", MODE_PRIVATE).getString("sensor_device_id", defaultDevice))
        tvSignedInUser.text = "Signed in — ${getSharedPreferences("fallguard_user", MODE_PRIVATE).getString("name", "FallGuard member")}"

        btnToggleMonitoring.setOnClickListener { if (isMonitoringActive) stopMonitoring() else startMonitoring() }
        btnLogout.setOnClickListener { logout() }
        btnUpdateContact.setOnClickListener { showEmergencyContactEditor() }
        btnSaveBackendUrl.setOnClickListener { saveBackendConfiguration(testConnection = false) }
        btnTestBackendUrl.setOnClickListener { saveBackendConfiguration(testConnection = true) }
    }

    override fun onStart() {
        super.onStart()
        lastKnownNetwork = networkManager.activeNetwork
        try {
            networkManager.registerDefaultNetworkCallback(backendNetworkCallback)
            networkCallbackRegistered = true
        } catch (error: Exception) {
            tvBackendConnectionResult.text = "Network change monitoring is unavailable: ${error.message ?: "permission or platform error"}"
        }
    }

    override fun onStop() {
        if (networkCallbackRegistered) {
            runCatching { networkManager.unregisterNetworkCallback(backendNetworkCallback) }
            networkCallbackRegistered = false
        }
        super.onStop()
    }

    private fun saveBackendConfiguration(testConnection: Boolean) {
        val previousUrl = BackendConfig.getBaseUrl(this)
        val newUrl = BackendConfig.saveBaseUrl(this, etBackendUrl.text?.toString().orEmpty())
        if (newUrl == null) {
            tvBackendConnectionResult.setTextColor(ContextCompat.getColor(this, R.color.danger))
            tvBackendConnectionResult.text = "Enter a valid URL: emulator http://10.0.2.2:8000, phone PC LAN URL, or production HTTPS URL."
            return
        }
        val changed = previousUrl != newUrl
        val reconnectMonitor = changed && isMonitoringActive
        val deviceId = etDeviceId.text?.toString()?.trim().orEmpty()
        if (reconnectMonitor) stopMonitoring()
        btnSaveBackendUrl.isEnabled = false
        btnTestBackendUrl.isEnabled = false
        tvBackendConnectionResult.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
        tvBackendConnectionResult.text = if (testConnection) "Checking backend health and session…" else "Saving backend address…"
        lifecycleScope.launch {
            try {
                val backendName = if (testConnection) authApi.testConnection(session.accessToken) else null
                if (reconnectMonitor) {
                    if (deviceId.length < 4) throw IllegalArgumentException("Device ID must contain at least four characters before monitoring can reconnect.")
                    authApi.registerDevice(session.accessToken, deviceId, "FallGuard Android phone")
                    startMonitoringService(deviceId)
                }
                tvBackendConnectionResult.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.success))
                tvBackendConnectionResult.text = when {
                    backendName != null -> "Connected to $backendName; session verified at $newUrl."
                    reconnectMonitor -> "Saved $newUrl and reconnected monitoring securely."
                    else -> "Saved backend URL: $newUrl"
                }
            } catch (error: ApiException) {
                tvBackendConnectionResult.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.danger))
                tvBackendConnectionResult.text = "Backend returned HTTP ${error.statusCode}: ${error.message}"
                if (error.statusCode == 401) expireSession("Your session is invalid or expired. Please sign in again.")
            } catch (error: Exception) {
                tvBackendConnectionResult.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.danger))
                tvBackendConnectionResult.text = error.message ?: "Could not save or test the backend address."
            } finally {
                btnSaveBackendUrl.isEnabled = true
                btnTestBackendUrl.isEnabled = true
            }
        }
    }

    private fun loadDashboardOnce() {
        if (!session.isValid()) {
            expireSession("Your session expired. Please sign in again.")
            return
        }
        lifecycleScope.launch {
            try {
                val profile = authApi.currentUser(session.accessToken)
                signedInProfile = profile
                tvSignedInUser.text = "Signed in — ${profile.name}"
                val dashboard = authApi.dashboard(session.accessToken)
                tvSignedInUser.text = "Signed in — ${dashboard.name}"
                getSharedPreferences("fallguard_user", MODE_PRIVATE).edit().putString("name", dashboard.name).apply()
                val safetyDetails = authApi.safetyDetails(session.accessToken)
                tvDashboardSummary.text = "Activity: ${dashboard.activity}\nRisk: ${dashboard.risk}  •  Status: ${dashboard.status}\nFalls recorded: ${dashboard.falls}  •  Activities: ${dashboard.activities}\n\n$safetyDetails"
            } catch (e: ApiException) {
                if (e.statusCode == 401) expireSession("Your session is invalid or expired. Please sign in again.")
                else tvDashboardSummary.text = "Dashboard unavailable (HTTP ${e.statusCode}): ${e.message}"
            } catch (e: Exception) {
                tvDashboardSummary.text = "Dashboard unavailable: ${e.message ?: "Check your backend URL and connection."}"
            }
        }
    }

    private fun showEmergencyContactEditor() {
        val profile = signedInProfile ?: run {
            tvDashboardSummary.text = "Load your authenticated profile before editing emergency contacts."
            return
        }
        val fields = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = (20 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad / 2, pad, 0)
        }
        fun input(hintText: String, value: String?, type: Int): EditText = EditText(this).also { edit ->
            edit.hint = hintText
            edit.setText(value.orEmpty())
            edit.inputType = type
            fields.addView(edit)
        }
        val name = input("Emergency contact name", profile.emergency_contact_name, InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS)
        val phone = input("Phone with country code (for example +15551234567)", profile.emergency_contact_phone, InputType.TYPE_CLASS_PHONE)
        val email = input("Emergency contact email", profile.emergency_contact_email, InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS)
        val dialog = AlertDialog.Builder(this)
            .setTitle("Emergency contact")
            .setView(fields)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Save", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val contactName = name.text.toString().trim()
                val contactPhone = phone.text.toString().trim()
                if (contactName.isBlank() || !contactPhone.matches(Regex("^\\+[1-9]\\d{7,14}$"))) {
                    tvDashboardSummary.text = "Enter an emergency contact name and an international phone number."
                    return@setOnClickListener
                }
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = false
                lifecycleScope.launch {
                    try {
                        authApi.updateEmergencyContact(session.accessToken, contactName, contactPhone, email.text.toString())
                        signedInProfile = authApi.currentUser(session.accessToken)
                        tvDashboardSummary.text = "Emergency contact saved. Refreshing your private dashboard…"
                        loadDashboardOnce()
                        dialog.dismiss()
                    } catch (e: ApiException) {
                        if (e.statusCode == 401) expireSession("Your session is invalid or expired. Please sign in again.")
                        else tvDashboardSummary.text = "Could not update emergency contact (HTTP ${e.statusCode}): ${e.message}"
                        dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = true
                    } catch (e: Exception) {
                        tvDashboardSummary.text = "Could not update emergency contact: ${e.message ?: "Check your backend connection."}"
                        dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = true
                    }
                }
            }
        }
        dialog.show()
    }

    private fun startMonitoring() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), permissionsRequestCode)
            tvDashboardSummary.text = "Allow notifications, then tap Start Monitoring again."
            return
        }
        val baseUrl = BackendConfig.saveBaseUrl(this, etBackendUrl.text?.toString().orEmpty())
        if (baseUrl == null) {
            tvDashboardSummary.text = "Enter a valid backend URL (for example http://10.0.2.2:8000)."
            return
        }
        val deviceId = etDeviceId.text?.toString()?.trim().orEmpty()
        if (deviceId.length < 4) {
            tvDashboardSummary.text = "Device ID must contain at least four characters."
            return
        }
        if (!session.isValid()) {
            expireSession("Your session expired. Please sign in again.")
            return
        }
        btnToggleMonitoring.isEnabled = false
        btnToggleMonitoring.text = "Connecting…"
        lifecycleScope.launch {
            try {
                // Register ownership with the authenticated API before opening the sensor socket.
                authApi.registerDevice(session.accessToken, deviceId, "FallGuard Android phone")
                getSharedPreferences("fallguard_settings", MODE_PRIVATE).edit().putString("sensor_device_id", deviceId).apply()
                startMonitoringService(deviceId)
                tvDashboardSummary.text = "Device registered. Connecting securely to sensor monitoring…"
                promptBackgroundOperationSettings()
            } catch (e: ApiException) {
                if (e.statusCode == 401) expireSession("Your session expired. Please sign in again.")
                else tvDashboardSummary.text = "Could not register this device (HTTP ${e.statusCode}): ${e.message}"
            } catch (e: Exception) {
                tvDashboardSummary.text = "Could not connect to backend: ${e.message ?: "Check the backend URL."}"
            } finally {
                btnToggleMonitoring.isEnabled = true
                if (!isMonitoringActive) btnToggleMonitoring.text = getString(R.string.btn_start)
            }
        }
    }

    private fun startMonitoringService(deviceId: String) {
        val serviceIntent = Intent(this, SensorForegroundService::class.java).apply {
            putExtra("SERVER_URL", BackendConfig.sensorWebSocketUrl(this@MainActivity))
            putExtra("DEVICE_ID", deviceId)
            putExtra("JWT_TOKEN", session.accessToken)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(serviceIntent) else startService(serviceIntent)
        if (!bound) bound = bindService(serviceIntent, connection, Context.BIND_AUTO_CREATE)
        isMonitoringActive = true
        btnToggleMonitoring.text = getString(R.string.btn_stop)
        btnToggleMonitoring.setBackgroundColor(ContextCompat.getColor(this, R.color.danger))
    }

    private fun stopMonitoring() {
        service?.webSocketClient?.onConnectionStateChanged = null
        service?.webSocketClient?.disconnect()
        val serviceIntent = Intent(this, SensorForegroundService::class.java)
        if (bound) {
            try { unbindService(connection) } catch (_: Exception) { }
            bound = false
        }
        stopService(serviceIntent)
        service = null
        isMonitoringActive = false
        btnToggleMonitoring.text = getString(R.string.btn_start)
        btnToggleMonitoring.setBackgroundColor(ContextCompat.getColor(this, R.color.primary))
        tvStatus.text = getString(R.string.status_disconnected)
        tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
        tvDashboardSummary.text = "Monitoring stopped. Your fall detection and emergency contacts remain configured."
    }

    private fun logout() {
        // Stop a surviving foreground service even if the screen was recreated and no longer bound.
        stopMonitoring()
        SecureSessionStore.clear(this)
        goToLogin("You have been signed out.")
    }

    private fun expireSession(message: String) {
        stopMonitoring()
        SecureSessionStore.clear(this)
        goToLogin(message)
    }

    private fun goToLogin(message: String) {
        startActivity(Intent(this, LoginActivity::class.java).putExtra("notice", message)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK))
        finish()
    }

    private fun requestAppPermissions() {
        val needed = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) needed.add(Manifest.permission.POST_NOTIFICATIONS)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) needed.add(Manifest.permission.ACCESS_FINE_LOCATION)
        if (needed.isNotEmpty()) ActivityCompat.requestPermissions(this, needed.toTypedArray(), permissionsRequestCode)
    }

    private fun showFirstRunPlacementInstructions() {
        val prefs = getSharedPreferences("fallguard_settings", MODE_PRIVATE)
        if (prefs.getBoolean("pocket_placement_acknowledged", false)) return
        AlertDialog.Builder(this)
            .setTitle("Before monitoring")
            .setMessage("Carry the phone securely in a trouser pocket, in the same orientation used during the MobiFall study. Keep the screen off during monitoring. This is a research prototype, not a medical device.")
            .setPositiveButton("Understood") { _, _ -> prefs.edit().putBoolean("pocket_placement_acknowledged", true).apply() }
            .setCancelable(false)
            .show()
    }

    private fun promptBackgroundOperationSettings() {
        val power = getSystemService(POWER_SERVICE) as PowerManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !power.isIgnoringBatteryOptimizations(packageName)) {
            val prefs = getSharedPreferences("fallguard_settings", MODE_PRIVATE)
            if (!prefs.getBoolean("battery_prompt_shown", false)) {
                prefs.edit().putBoolean("battery_prompt_shown", true).apply()
                AlertDialog.Builder(this)
                    .setTitle("Allow background monitoring")
                    .setMessage("Android may pause sensor streaming when the screen is off. For a research session, open Battery Optimization settings and allow FallGuard to run unrestricted. This increases battery use.")
                    .setNegativeButton("Later", null)
                    .setPositiveButton("Open settings") { _, _ ->
                        runCatching { startActivity(Intent(android.provider.Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)) }
                    }
                    .show()
            }
        }
        if (Build.VERSION.SDK_INT >= 34) {
            val manager = getSystemService(NotificationManager::class.java)
            if (!manager.canUseFullScreenIntent()) {
                AlertDialog.Builder(this)
                    .setTitle("Enable fall alert display")
                    .setMessage("Android needs full-screen notification access to show a fall confirmation over the lock screen. You can enable FallGuard in the next screen.")
                    .setNegativeButton("Later", null)
                    .setPositiveButton("Open settings") { _, _ ->
                        runCatching { startActivity(Intent(android.provider.Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT).setData(android.net.Uri.parse("package:$packageName"))) }
                    }
                    .show()
            }
        }
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(statsRunnable)
        service?.webSocketClient?.onConnectionStateChanged = null
        if (bound) try { unbindService(connection) } catch (_: Exception) { }
        super.onDestroy()
    }
}
