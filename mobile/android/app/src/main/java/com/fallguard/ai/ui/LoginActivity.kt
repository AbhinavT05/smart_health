package com.fallguard.ai.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.fallguard.ai.R
import com.fallguard.ai.auth.ApiException
import com.fallguard.ai.auth.AuthApi
import com.fallguard.ai.auth.BackendConfig
import com.fallguard.ai.auth.SecureSessionStore
import com.fallguard.ai.service.SensorForegroundService
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.launch

class LoginActivity : AppCompatActivity() {
    private lateinit var email: TextInputEditText
    private lateinit var password: TextInputEditText
    private lateinit var backendUrl: TextInputEditText
    private lateinit var error: TextView
    private lateinit var progress: ProgressBar
    private lateinit var loginButton: MaterialButton
    private lateinit var authApi: AuthApi

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)
        email = findViewById(R.id.et_login_email)
        password = findViewById(R.id.et_login_password)
        backendUrl = findViewById(R.id.et_backend_url)
        error = findViewById(R.id.tv_login_error)
        progress = findViewById(R.id.pb_login)
        loginButton = findViewById(R.id.btn_login)
        backendUrl.setText(BackendConfig.getBaseUrl(this))
        authApi = AuthApi(this)

        val session = SecureSessionStore.read(this)
        if (session != null) {
            setLoading(true, "Checking saved session…")
            lifecycleScope.launch {
                try {
                    authApi.currentUser(session.accessToken)
                    openDashboard()
                } catch (e: ApiException) {
                    SecureSessionStore.clear(this@LoginActivity)
                    stopService(Intent(this@LoginActivity, SensorForegroundService::class.java))
                    showError(if (e.statusCode == 401) "Your session expired. Please sign in again." else e.message)
                    setLoading(false)
                } catch (e: Exception) {
                    showError("Could not verify your saved session. Check the backend URL and connection, then sign in.")
                    setLoading(false)
                }
            }
        } else {
            stopService(Intent(this, SensorForegroundService::class.java))
            setLoading(false)
            intent.getStringExtra("notice")?.let(::showError)
        }

        loginButton.setOnClickListener { submitLogin() }
        findViewById<TextView>(R.id.tv_create_account).setOnClickListener {
            startActivity(Intent(this, SignupActivity::class.java))
        }
    }

    private fun submitLogin() {
        val cleanEmail = email.text?.toString()?.trim().orEmpty()
        val secret = password.text?.toString().orEmpty()
        if (!cleanEmail.matches(Regex("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$"))) {
            showError("Enter a valid email address.")
            email.requestFocus()
            return
        }
        if (secret.isBlank()) {
            showError("Enter your password.")
            password.requestFocus()
            return
        }
        if (BackendConfig.saveBaseUrl(this, backendUrl.text?.toString().orEmpty()) == null) {
            showError("Enter a valid backend URL such as http://10.0.2.2:8000 or https://api.example.com.")
            backendUrl.requestFocus()
            return
        }
        showError(null)
        setLoading(true)
        lifecycleScope.launch {
            try {
                val result = authApi.login(cleanEmail, secret)
                SecureSessionStore.save(this@LoginActivity, result.access_token, result.expires_in_minutes)
                getSharedPreferences("fallguard_user", MODE_PRIVATE).edit().putString("name", result.user?.name).apply()
                openDashboard()
            } catch (e: Exception) {
                showError(e.message ?: "Sign in failed. Please try again.")
                setLoading(false)
            }
        }
    }

    private fun openDashboard() {
        startActivity(Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK))
        finish()
    }

    private fun setLoading(loading: Boolean, label: String? = null) {
        progress.visibility = if (loading) View.VISIBLE else View.GONE
        loginButton.isEnabled = !loading
        loginButton.text = label ?: if (loading) "Signing In…" else "Sign In"
    }

    private fun showError(message: String?) {
        error.text = message.orEmpty()
        error.visibility = if (message.isNullOrBlank()) View.GONE else View.VISIBLE
    }
}
