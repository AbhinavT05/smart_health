package com.fallguard.ai.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.fallguard.ai.R
import com.fallguard.ai.auth.AuthApi
import com.fallguard.ai.auth.BackendConfig
import com.fallguard.ai.auth.SecureSessionStore
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.launch

class SignupActivity : AppCompatActivity() {
    private lateinit var name: TextInputEditText
    private lateinit var email: TextInputEditText
    private lateinit var password: TextInputEditText
    private lateinit var phone: TextInputEditText
    private lateinit var emergencyName: TextInputEditText
    private lateinit var emergencyPhone: TextInputEditText
    private lateinit var emergencyEmail: TextInputEditText
    private lateinit var backendUrl: TextInputEditText
    private lateinit var error: TextView
    private lateinit var progress: ProgressBar
    private lateinit var registerButton: MaterialButton
    private lateinit var testBackendButton: MaterialButton
    private lateinit var backendTestStatus: TextView
    private lateinit var authApi: AuthApi

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_signup)
        name = findViewById(R.id.et_signup_name)
        email = findViewById(R.id.et_signup_email)
        password = findViewById(R.id.et_signup_password)
        phone = findViewById(R.id.et_signup_phone)
        emergencyName = findViewById(R.id.et_signup_emergency_name)
        emergencyPhone = findViewById(R.id.et_signup_emergency_phone)
        emergencyEmail = findViewById(R.id.et_signup_emergency_email)
        backendUrl = findViewById(R.id.et_signup_backend_url)
        error = findViewById(R.id.tv_signup_error)
        progress = findViewById(R.id.pb_signup)
        registerButton = findViewById(R.id.btn_signup)
        testBackendButton = findViewById(R.id.btn_test_backend)
        backendTestStatus = findViewById(R.id.tv_backend_test)
        backendUrl.setText(BackendConfig.getBaseUrl(this))
        authApi = AuthApi(this)
        registerButton.setOnClickListener { submitRegistration() }
        testBackendButton.setOnClickListener { testBackend() }
        findViewById<TextView>(R.id.tv_back_to_login).setOnClickListener { finish() }
    }

    private fun testBackend() {
        val baseUrl = BackendConfig.saveBaseUrl(this, backendUrl.value())
        if (baseUrl == null) {
            backendTestStatus.setTextColor(getColor(R.color.danger))
            backendTestStatus.text = "Enter a valid backend URL. Emulator: http://10.0.2.2:8000; physical phone: PC LAN URL; production: HTTPS."
            return
        }
        testBackendButton.isEnabled = false
        testBackendButton.text = "Testing…"
        backendTestStatus.setTextColor(getColor(R.color.text_secondary))
        backendTestStatus.text = "Checking /api/health…"
        lifecycleScope.launch {
            try {
                val backendName = authApi.testConnection()
                backendTestStatus.setTextColor(getColor(R.color.success))
                backendTestStatus.text = "Connected to $backendName at $baseUrl."
            } catch (error: Exception) {
                backendTestStatus.setTextColor(getColor(R.color.danger))
                backendTestStatus.text = error.message ?: "Backend connection test failed."
            } finally {
                testBackendButton.isEnabled = true
                testBackendButton.text = "Test backend connection"
            }
        }
    }

    private fun submitRegistration() {
        val fullName = name.value()
        val emailValue = email.value()
        val passwordValue = password.value()
        val phoneValue = phone.value()
        val contactName = emergencyName.value()
        val contactPhone = emergencyPhone.value()
        val contactEmail = emergencyEmail.value()
        when {
            fullName.length < 2 -> return showError("Enter your full name (at least 2 characters).")
            !emailValue.matches(Regex("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$")) -> return showError("Enter a valid email address.")
            passwordValue.length < 8 -> return showError("Password must be at least 8 characters.")
            !phoneValue.matches(Regex("^\\+[1-9]\\d{7,14}$")) -> return showError("Enter your phone with country code, such as +15551234567.")
            contactName.isBlank() -> return showError("Enter an emergency contact name.")
            !contactPhone.matches(Regex("^\\+[1-9]\\d{7,14}$")) -> return showError("Enter the emergency phone with country code, such as +15551234567.")
            !contactEmail.matches(Regex("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$")) -> return showError("Enter a valid emergency contact email.")
        }
        if (BackendConfig.saveBaseUrl(this, backendUrl.value()) == null) {
            showError("Enter a valid backend URL such as http://10.0.2.2:8000 or https://api.example.com.")
            return
        }
        showError(null)
        setLoading(true)
        lifecycleScope.launch {
            try {
                val result = authApi.register(fullName, emailValue, passwordValue, phoneValue, contactName, contactPhone, contactEmail)
                SecureSessionStore.save(this@SignupActivity, result.access_token, result.expires_in_minutes)
                getSharedPreferences("fallguard_user", MODE_PRIVATE).edit().putString("name", result.user?.name).apply()
                startActivity(Intent(this@SignupActivity, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK))
                finish()
            } catch (e: Exception) {
                showError(e.message ?: "Registration failed. Please check your details and try again.")
                setLoading(false)
            }
        }
    }

    private fun TextInputEditText.value() = text?.toString()?.trim().orEmpty()

    private fun setLoading(loading: Boolean) {
        progress.visibility = if (loading) View.VISIBLE else View.GONE
        registerButton.isEnabled = !loading
        registerButton.text = if (loading) "Creating Account…" else "Create Account"
    }

    private fun showError(message: String?) {
        error.text = message.orEmpty()
        error.visibility = if (message.isNullOrBlank()) View.GONE else View.VISIBLE
    }
}
