package com.fallguard.ai.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.fallguard.ai.R
import com.fallguard.ai.location.LocationHelper
import com.fallguard.ai.service.SensorForegroundService
import com.google.android.material.button.MaterialButton

/**
 * High-Priority Fall Verification Screen.
 * Wakes the display and sounds/vibrates an alert.
 * Allows the user to either:
 * 1. Cancel false alarms via "I'M OK"
 * 2. Immediately escalate via "NEED HELP"
 * 3. Fall back to automatic emergency dispatch upon countdown expiry.
 */
class FallConfirmationActivity : AppCompatActivity() {

    private var eventId: Int = 0
    private var responseSubmitted = false
    private var timeoutSeconds: Float = 15f
    private var countDownTimer: CountDownTimer? = null
    private var service: SensorForegroundService? = null
    private lateinit var locationHelper: LocationHelper

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as? SensorForegroundService.LocalBinder)?.getService()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Wake and show over lock screen
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
            )
        }

        setContentView(R.layout.activity_fall_confirmation)

        eventId = intent.getIntExtra("EVENT_ID", 0)
        timeoutSeconds = intent.getFloatExtra("TIMEOUT_SECONDS", 15f)
        locationHelper = LocationHelper(this)

        bindService(
            Intent(this, SensorForegroundService::class.java),
            connection,
            Context.BIND_AUTO_CREATE
        )

        triggerHapticAlert()
        initViews()
    }

    private fun initViews() {
        val tvCountdown = findViewById<TextView>(R.id.tv_countdown)
        val btnImOk = findViewById<MaterialButton>(R.id.btn_im_ok)
        val btnNeedHelp = findViewById<MaterialButton>(R.id.btn_need_help)

        // Start visual countdown timer
        val durationMillis = (timeoutSeconds * 1000).toLong()
        countDownTimer = object : CountDownTimer(durationMillis, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                val secondsLeft = millisUntilFinished / 1000
                tvCountdown.text = secondsLeft.toString()
            }

            override fun onFinish() {
                tvCountdown.text = "0"
                // Time expired: backend automatically escalates to emergency contact
                finish()
            }
        }.start()

        btnImOk.setOnClickListener {
            handleConfirmationAction("I_AM_OK")
        }

        btnNeedHelp.setOnClickListener {
            handleConfirmationAction("NEED_HELP")
        }
    }

    private fun handleConfirmationAction(action: String) {
        if (responseSubmitted) return
        responseSubmitted = true
        countDownTimer?.cancel()
        locationHelper.getEmergencyLocation { loc ->
            service?.submitFallConfirmation(eventId, action, loc)
            service?.dismissFallNotification(eventId)
            finish()
        }
    }

    private fun triggerHapticAlert() {
        try {
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val manager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                manager.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }

            val pattern = longArrayOf(0, 500, 200, 500, 200, 800)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(pattern, -1)
            }
        } catch (e: Exception) {
            // Ignore if vibration unavailable
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        countDownTimer?.cancel()
        try {
            unbindService(connection)
        } catch (e: Exception) {
            // Ignore
        }
    }
}
