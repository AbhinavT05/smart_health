package com.fallguard.ai.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.os.IBinder
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.fallguard.ai.R
import com.fallguard.ai.location.LocationHelper
import com.fallguard.ai.model.FallResolution
import com.fallguard.ai.service.SensorForegroundService
import com.google.android.material.button.MaterialButton

/** Full-screen fall confirmation UI launched only through Android's full-screen notification path. */
class FallAlertActivity : AppCompatActivity() {
    private var eventId: Int = 0
    private var deadlineElapsed: Long = 0L
    private var responseSubmitted = false
    private var awaitingBackend = false
    private var countdownExpired = false
    private var backendWaitDeadline = 0L
    private var backendRetryAllowed = false
    private val mainHandler = Handler(Looper.getMainLooper())
    private var service: SensorForegroundService? = null
    private var countDownTimer: CountDownTimer? = null
    private var alarmPlayer: MediaPlayer? = null
    private var alertingActive = false
    private var vibrator: Vibrator? = null
    private var bound = false
    private val backendWaitExpired = Runnable {
        if (awaitingBackend && responseSubmitted && backendWaitDeadline > 0L) {
            awaitingBackend = false
            backendWaitDeadline = 0L
            if (backendRetryAllowed) {
                responseSubmitted = false
                backendRetryAllowed = false
                setButtonsEnabled(true)
                status.text = getString(R.string.fall_alert_network_error)
            } else {
                status.text = getString(R.string.fall_alert_expiry_unconfirmed)
            }
        }
    }

    private lateinit var countdown: TextView
    private lateinit var status: TextView
    private lateinit var okayButton: MaterialButton
    private lateinit var helpButton: MaterialButton

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as? SensorForegroundService.LocalBinder)?.getService()
            service?.setFallAlertListener(eventId) { result -> runOnUiThread { showResolution(result) } }
            if (awaitingBackend) status.text = getString(R.string.fall_alert_waiting_backend)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            if (awaitingBackend) status.text = getString(R.string.fall_alert_network_error)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_fall_confirmation)

        eventId = intent.getIntExtra(EXTRA_EVENT_ID, 0)
        deadlineElapsed = savedInstanceState?.getLong(STATE_DEADLINE)
            ?: intent.getLongExtra(EXTRA_DEADLINE, 0L)
        responseSubmitted = savedInstanceState?.getBoolean(STATE_SUBMITTED) ?: false
        awaitingBackend = savedInstanceState?.getBoolean(STATE_AWAITING) ?: false
        countdownExpired = savedInstanceState?.getBoolean(STATE_COUNTDOWN_EXPIRED) ?: false
        backendWaitDeadline = savedInstanceState?.getLong(STATE_BACKEND_DEADLINE) ?: 0L
        backendRetryAllowed = savedInstanceState?.getBoolean(STATE_RETRY_ALLOWED) ?: false
        if (eventId <= 0 || deadlineElapsed <= 0L) {
            finish()
            return
        }

        countdown = findViewById(R.id.tv_countdown)
        status = findViewById(R.id.tv_fall_alert_status)
        okayButton = findViewById(R.id.btn_im_ok)
        helpButton = findViewById(R.id.btn_need_help)
        okayButton.setOnClickListener { submitResponse(ACTION_OK) }
        helpButton.setOnClickListener { submitResponse(ACTION_HELP) }

        val bindIntent = Intent(this, SensorForegroundService::class.java)
        bound = bindService(bindIntent, connection, Context.BIND_AUTO_CREATE)
        if (!awaitingBackend && !responseSubmitted) {
            if (deadlineElapsed <= SystemClock.elapsedRealtime()) {
                onCountdownExpired()
            } else {
                startCountdown()
                startAlarm()
                triggerHapticAlert()
            }
        } else {
            val waitExpired = backendWaitDeadline > 0L && backendWaitDeadline <= SystemClock.elapsedRealtime()
            val canRetry = waitExpired && backendRetryAllowed
            if (canRetry) {
                responseSubmitted = false
                awaitingBackend = false
                backendRetryAllowed = false
                backendWaitDeadline = 0L
            }
            setButtonsEnabled(canRetry)
            status.text = if (waitExpired && countdownExpired) getString(R.string.fall_alert_expiry_unconfirmed)
            else if (waitExpired) getString(R.string.fall_alert_network_error)
            else getString(R.string.fall_alert_waiting_backend)
            if (backendWaitDeadline > SystemClock.elapsedRealtime()) {
                mainHandler.postDelayed(backendWaitExpired, backendWaitDeadline - SystemClock.elapsedRealtime())
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        val incomingEventId = intent.getIntExtra(EXTRA_EVENT_ID, 0)
        if (incomingEventId == eventId) return

        // Android may reuse a singleTop alert Activity for a newer notification.
        // Replace the old event cleanly so its timer/alarm/listener cannot leak.
        finish()
        startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }

    @Deprecated("Deprecated in AndroidX; the active alert must remain visible until resolved")
    override fun onBackPressed() {
        // Keep the emergency prompt and its ongoing notification active until a
        // backend resolution is received. The user can still use either action.
    }

    private fun startCountdown() {
        val remaining = (deadlineElapsed - SystemClock.elapsedRealtime()).coerceAtLeast(0L)
        if (remaining == 0L) {
            onCountdownExpired()
            return
        }
        countDownTimer?.cancel()
        countDownTimer = object : CountDownTimer(remaining, 1000L) {
            override fun onTick(millisUntilFinished: Long) {
                countdown.text = (millisUntilFinished / 1000L).coerceAtLeast(1L).toString()
            }
            override fun onFinish() = onCountdownExpired()
        }.start()
    }

    private fun onCountdownExpired() {
        countdown.text = "0"
        countdownExpired = true
        responseSubmitted = true
        awaitingBackend = true
        backendWaitDeadline = SystemClock.elapsedRealtime() + BACKEND_ACK_TIMEOUT_MS
        backendRetryAllowed = false
        mainHandler.removeCallbacks(backendWaitExpired)
        mainHandler.postDelayed(backendWaitExpired, BACKEND_ACK_TIMEOUT_MS)
        setButtonsEnabled(false)
        stopAlerting()
        status.text = getString(R.string.fall_alert_waiting_backend)
    }

    private fun submitResponse(action: String) {
        if (responseSubmitted) return
        responseSubmitted = true
        awaitingBackend = true
        backendRetryAllowed = false
        countDownTimer?.cancel()
        stopAlerting()
        setButtonsEnabled(false)
        status.text = getString(R.string.fall_alert_sending)

        LocationHelper(this).getEmergencyLocation { location ->
            runOnUiThread {
                val sent = service?.submitFallConfirmation(eventId, action, location) == true
                if (!sent) {
                    responseSubmitted = false
                    awaitingBackend = false
                    setButtonsEnabled(true)
                    status.text = getString(R.string.fall_alert_network_error)
                } else {
                    backendRetryAllowed = true
                    backendWaitDeadline = SystemClock.elapsedRealtime() + BACKEND_ACK_TIMEOUT_MS
                    mainHandler.removeCallbacks(backendWaitExpired)
                    mainHandler.postDelayed(backendWaitExpired, BACKEND_ACK_TIMEOUT_MS)
                }
            }
        }
    }

    private fun showResolution(result: FallResolution) {
        if (result.eventId != eventId) return
        countDownTimer?.cancel()
        stopAlerting()
        responseSubmitted = true
        awaitingBackend = false
        backendWaitDeadline = 0L
        backendRetryAllowed = false
        mainHandler.removeCallbacks(backendWaitExpired)
        setButtonsEnabled(false)
        status.text = when (result.status) {
            "CANCELLED_BY_USER" -> getString(R.string.fall_alert_cancelled)
            "CONFIRMED" -> escalationResultText(result)
            else -> "Server response: ${result.status.ifBlank { "unknown" }}"
        }
        window.decorView.postDelayed({ if (!isFinishing) finish() }, 3500L)
    }

    private fun escalationResultText(result: FallResolution): String {
        if (result.alertDispatchRequested != true) {
            return getString(R.string.fall_alert_no_contacts)
        }
        if (result.simulated) return "Emergency workflow was processed in simulation mode. No message delivery is claimed."
        val statuses = result.deliveryStatuses.distinct().joinToString(", ")
        return if (statuses.isBlank()) {
            "Emergency workflow was submitted. Contact delivery is not confirmed."
        } else {
            "Emergency workflow was submitted. Provider status: $statuses. This does not confirm delivery."
        }
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        okayButton.isEnabled = enabled
        helpButton.isEnabled = enabled
    }

    private fun startAlarm() {
        runCatching {
            val alarmUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            if (alarmUri == null) return
            alertingActive = true
            alarmPlayer = MediaPlayer().apply {
                setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ALARM)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build())
                isLooping = true
                setDataSource(this@FallAlertActivity, alarmUri)
                setOnPreparedListener { player ->
                    if (alertingActive && !isFinishing) player.start() else player.release()
                }
                setOnErrorListener { player, _, _ ->
                    if (alarmPlayer === player) alarmPlayer = null
                    player.release()
                    true
                }
                prepareAsync()
            }
        }
    }

    private fun stopAlerting() {
        countDownTimer?.cancel()
        alertingActive = false
        alarmPlayer?.let { player ->
            runCatching { if (player.isPlaying) player.stop() }
            runCatching { player.release() }
        }
        alarmPlayer = null
        runCatching { vibrator?.cancel() }
    }

    private fun triggerHapticAlert() {
        runCatching {
            vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }
            val pattern = longArrayOf(0, 500, 200, 500, 200, 800)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator?.vibrate(VibrationEffect.createWaveform(pattern, 0))
            } else {
                @Suppress("DEPRECATION")
                vibrator?.vibrate(pattern, 0)
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putLong(STATE_DEADLINE, deadlineElapsed)
        outState.putBoolean(STATE_SUBMITTED, responseSubmitted)
        outState.putBoolean(STATE_AWAITING, awaitingBackend)
        outState.putBoolean(STATE_COUNTDOWN_EXPIRED, countdownExpired)
        outState.putLong(STATE_BACKEND_DEADLINE, backendWaitDeadline)
        outState.putBoolean(STATE_RETRY_ALLOWED, backendRetryAllowed)
        super.onSaveInstanceState(outState)
    }

    override fun onDestroy() {
        stopAlerting()
        mainHandler.removeCallbacks(backendWaitExpired)
        service?.clearFallAlertListener(eventId)
        if (bound) runCatching { unbindService(connection) }
        super.onDestroy()
    }

    companion object {
        const val EXTRA_EVENT_ID = "EVENT_ID"
        const val EXTRA_DEADLINE = "DEADLINE_ELAPSED"
        private const val STATE_DEADLINE = "state_deadline_elapsed"
        private const val STATE_SUBMITTED = "state_response_submitted"
        private const val STATE_AWAITING = "state_awaiting_backend"
        private const val STATE_COUNTDOWN_EXPIRED = "state_countdown_expired"
        private const val STATE_BACKEND_DEADLINE = "state_backend_wait_deadline"
        private const val STATE_RETRY_ALLOWED = "state_backend_retry_allowed"
        private const val BACKEND_ACK_TIMEOUT_MS = 15_000L
        private const val ACTION_OK = "I_AM_OK"
        private const val ACTION_HELP = "NEED_HELP"
    }
}
