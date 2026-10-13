package com.fallguard.ai.model

import com.google.gson.annotations.SerializedName

/**
 * Fall confirmation response sent by user back to backend.
 */
data class FallConfirmationResponse(
    @SerializedName("type") val type: String = "fall_confirmation",
    @SerializedName("event_id") val eventId: Int,
    @SerializedName("action") val action: String, // "I_AM_OK" or "NEED_HELP"
    @SerializedName("location") val location: Map<String, Double>? = null
)

/**
 * Incoming event from backend requesting fall confirmation.
 */
data class IncomingFallEvent(
    @SerializedName("id") val id: Int,
    @SerializedName("event_id") val eventId: Int,
    @SerializedName("device_id") val deviceId: String,
    @SerializedName("confidence") val confidence: Float,
    @SerializedName("risk_level") val riskLevel: String,
    @SerializedName("timeout_seconds") val timeoutSeconds: Float,
    @SerializedName("safety_override") val safetyOverride: Boolean,
    @SerializedName("message") val message: String
)

data class IncomingPendingFallMessage(
    @SerializedName("type") val type: String,
    @SerializedName("event") val event: IncomingFallEvent?
)

data class FallResolution(
    val eventId: Int,
    val status: String,
    val alertDispatchRequested: Boolean? = null,
    val simulated: Boolean = false,
    val deliveryStatuses: List<String> = emptyList(),
)
