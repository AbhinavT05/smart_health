package com.fallguard.ai.auth

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

data class AuthSession(val accessToken: String, val expiresAtMillis: Long) {
    fun isValid(now: Long = System.currentTimeMillis()): Boolean = accessToken.isNotBlank() && expiresAtMillis > now + 30_000
}

/** Stores the bearer token encrypted with a non-exportable Android Keystore AES key. */
object SecureSessionStore {
    private const val PREFS = "fallguard_secure_session"
    private const val VALUE = "encrypted_session"
    private const val KEY_ALIAS = "fallguard_session_aes_gcm_v1"

    fun save(context: Context, token: String, expiresInMinutes: Int) {
        require(token.isNotBlank())
        val expiresAt = System.currentTimeMillis() + expiresInMinutes.coerceAtLeast(1) * 60_000L
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val ciphertext = cipher.doFinal("$expiresAt\n$token".toByteArray(Charsets.UTF_8))
        val packed = cipher.iv + ciphertext
        val encoded = Base64.encodeToString(packed, Base64.NO_WRAP)
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(VALUE, encoded).apply()
    }

    fun read(context: Context): AuthSession? {
        val encoded = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(VALUE, null) ?: return null
        return try {
            val packed = Base64.decode(encoded, Base64.NO_WRAP)
            require(packed.size > 12)
            val iv = packed.copyOfRange(0, 12)
            val encrypted = packed.copyOfRange(12, packed.size)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), GCMParameterSpec(128, iv))
            val decoded = cipher.doFinal(encrypted).toString(Charsets.UTF_8)
            val expiresAt = decoded.substringBefore('\n').toLong()
            val token = decoded.substringAfter('\n')
            AuthSession(token, expiresAt).takeIf { it.isValid() }
        } catch (_: Exception) {
            clear(context)
            null
        }
    }

    fun clear(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(VALUE).apply()
    }

    private fun getOrCreateKey(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (keyStore.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(KEY_ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        return generator.generateKey()
    }
}
