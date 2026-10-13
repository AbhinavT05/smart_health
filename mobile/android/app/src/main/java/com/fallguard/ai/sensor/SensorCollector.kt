package com.fallguard.ai.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.fallguard.ai.model.RawSensorReading
import java.util.ArrayDeque
import java.util.concurrent.ConcurrentLinkedDeque
import kotlin.math.ceil
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/** Pairs independent Android accel/gyro clocks and linearly resamples onto 50 Hz. */
class SensorCollector(
    context: Context,
    private val maxBufferSamples: Int = 250,
    private val targetHz: Double = 50.0,
) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val accQueue = ArrayDeque<Point>()
    private val gyroQueue = ArrayDeque<Point>()
    private data class Point(val t: Long, val x: Float, val y: Float, val z: Float)
    private val accFilter = CausalButterworth4(targetHz)
    private val gyroFilter = CausalButterworth4(targetHz)

    val rollingBuffer = ConcurrentLinkedDeque<RawSensorReading>()
    @Volatile var latestAx = 0f; private set
    @Volatile var latestAy = 0f; private set
    @Volatile var latestAz = 9.81f; private set
    @Volatile var latestGx = 0f; private set
    @Volatile var latestGy = 0f; private set
    @Volatile var latestGz = 0f; private set
    @Volatile var measuredAccelHz = 0.0; private set
    @Volatile var measuredGyroHz = 0.0; private set
    @Volatile var emittedHz = 0.0; private set
    @Volatile var droppedGridSamples = 0L; private set
    @Volatile var sourceAccelEvents = 0L; private set
    @Volatile var sourceGyroEvents = 0L; private set

    var onReadingListener: ((RawSensorReading) -> Unit)? = null
    var onRateUpdate: ((String) -> Unit)? = null
    var isCollecting = false
        private set

    private val periodNs = (1_000_000_000.0 / targetHz).toLong()
    private var nextGridNs: Long? = null
    private var firstLogNs = 0L
    private var accelWindowCount = 0
    private var gyroWindowCount = 0
    private var outputWindowCount = 0
    private var lastRateLogNs = 0L
    private var lastAccelNs = 0L
    private var lastGyroNs = 0L
    private var lastWindowAccelNs = 0L
    private var lastWindowGyroNs = 0L
    private var lastWindowGridNs = 0L

    fun start() {
        if (isCollecting) return
        check(accelerometer != null) { "This phone has no accelerometer." }
        check(gyroscope != null) { "This phone has no gyroscope." }
        rollingBuffer.clear()
        accQueue.clear(); gyroQueue.clear(); nextGridNs = null
        accFilter.reset(); gyroFilter.reset()
        firstLogNs = 0L; lastRateLogNs = 0L
        lastAccelNs = 0L; lastGyroNs = 0L
        accelWindowCount = 0; gyroWindowCount = 0; outputWindowCount = 0
        droppedGridSamples = 0L; sourceAccelEvents = 0L; sourceGyroEvents = 0L
        // Collect above target rate; the shared grid below performs the explicit 50 Hz reduction.
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST)
        isCollecting = true
    }

    fun stop() {
        if (!isCollecting) return
        sensorManager.unregisterListener(this)
        accQueue.clear(); gyroQueue.clear(); rollingBuffer.clear()
        isCollecting = false
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || !isCollecting) return
        val t = event.timestamp // monotonic elapsed-realtime nanoseconds supplied by the sensor HAL
        if (firstLogNs == 0L) firstLogNs = t
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            sourceAccelEvents++; accelWindowCount++
            if (lastAccelNs != 0L) measuredAccelHz = 1e9 / (t - lastAccelNs).toDouble()
            lastAccelNs = t
            accQueue.addLast(accFilter.process(t, event.values, accelerometer?.minDelay ?: 0))
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            sourceGyroEvents++; gyroWindowCount++
            if (lastGyroNs != 0L) measuredGyroHz = 1e9 / (t - lastGyroNs).toDouble()
            lastGyroNs = t
            gyroQueue.addLast(gyroFilter.process(t, event.values, gyroscope?.minDelay ?: 0))
        } else return
        if (accQueue.isEmpty() || gyroQueue.isEmpty()) return
        if (nextGridNs == null) {
            val start = maxOf(accQueue.first.t, gyroQueue.first.t)
            nextGridNs = ceil(start.toDouble() / periodNs).toLong() * periodNs
        }
        drainReadyGrid()
        logRatesIfDue(t)
    }

    private fun drainReadyGrid() {
        var target = nextGridNs ?: return
        while (accQueue.size >= 2 && gyroQueue.size >= 2 && accQueue.last.t >= target && gyroQueue.last.t >= target) {
            val acc = interpolate(accQueue, target)
            val gyro = interpolate(gyroQueue, target)
            if (acc == null || gyro == null) {
                droppedGridSamples++
            } else {
                val reading = RawSensorReading(
                    timestamp = target / 1_000_000_000.0,
                    ax = acc.x, ay = acc.y, az = acc.z,
                    gx = gyro.x, gy = gyro.y, gz = gyro.z,
                )
                latestAx = reading.ax; latestAy = reading.ay; latestAz = reading.az
                latestGx = reading.gx; latestGy = reading.gy; latestGz = reading.gz
                rollingBuffer.addLast(reading)
                while (rollingBuffer.size > maxBufferSamples) rollingBuffer.pollFirst()
                outputWindowCount++
                onReadingListener?.invoke(reading)
            }
            target += periodNs
            nextGridNs = target
            pruneBeforeBracket(accQueue, target)
            pruneBeforeBracket(gyroQueue, target)
        }
        val oldest = maxOf(accQueue.firstOrNull()?.t ?: target, gyroQueue.firstOrNull()?.t ?: target)
        if (target < oldest) {
            val skipped = ((oldest - target + periodNs - 1) / periodNs).coerceAtLeast(0)
            droppedGridSamples += skipped
            nextGridNs = target + skipped * periodNs
        }
    }

    private fun interpolate(queue: ArrayDeque<Point>, target: Long): Point? {
        val points = queue.iterator()
        if (!points.hasNext()) return null
        var before = points.next()
        while (points.hasNext()) {
            val after = points.next()
            if (after.t >= target) {
                val gap = after.t - before.t
                if (gap <= 0L || gap > 100_000_000L || target < before.t) return null
                val fraction = (target - before.t).toDouble() / gap.toDouble()
                return Point(target,
                    (before.x + (after.x - before.x) * fraction).toFloat(),
                    (before.y + (after.y - before.y) * fraction).toFloat(),
                    (before.z + (after.z - before.z) * fraction).toFloat())
            }
            before = after
        }
        return null
    }

    private fun pruneBeforeBracket(queue: ArrayDeque<Point>, target: Long) {
        while (queue.size > 2 && queue.elementAt(1).t <= target) queue.removeFirst()
        while (queue.size > 600) queue.removeFirst()
    }

    private fun logRatesIfDue(nowNs: Long) {
        if (lastRateLogNs == 0L) lastRateLogNs = nowNs
        if (nowNs - lastRateLogNs < 10_000_000_000L) return
        val seconds = (nowNs - lastRateLogNs) / 1e9
        val accHz = accelWindowCount / seconds
        val gyroHz = gyroWindowCount / seconds
        val outHz = outputWindowCount / seconds
        measuredAccelHz = accHz; measuredGyroHz = gyroHz; emittedHz = outHz
        val message = "sensor_rates_hz accel=%.1f gyro=%.1f output=%.1f; dropped_grid=%d; source_events=%d".format(
            accHz, gyroHz, outHz, droppedGridSamples, (sourceAccelEvents + sourceGyroEvents).coerceAtLeast(0L))
        Log.i(TAG, message)
        onRateUpdate?.invoke(message)
        accelWindowCount = 0; gyroWindowCount = 0; outputWindowCount = 0
        lastRateLogNs = nowNs
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /** Streaming 4th-order Butterworth low-pass, matched cutoff/order to the offline loader. */
    private class CausalButterworth4(private val targetHz: Double) {
        private data class Section(val b0: Double, val b1: Double, val b2: Double, val a1: Double, val a2: Double)
        private var sampleRate = 0.0
        private var sections: List<Section> = emptyList()
        private val z1 = Array(2) { DoubleArray(3) }
        private val z2 = Array(2) { DoubleArray(3) }
        private var lastTimestamp = 0L

        fun reset() { sampleRate = 0.0; sections = emptyList(); lastTimestamp = 0L
            for (i in 0..1) { z1[i].fill(0.0); z2[i].fill(0.0) } }

        fun process(timestamp: Long, values: FloatArray, minDelayUs: Int): Point {
            val measuredHz = if (lastTimestamp > 0L && timestamp > lastTimestamp)
                1e9 / (timestamp - lastTimestamp).toDouble()
            else if (minDelayUs > 0) 1e6 / minDelayUs else targetHz * 4
            if (sampleRate == 0.0 || abs(measuredHz - sampleRate) / sampleRate > 0.10) configure(measuredHz)
            lastTimestamp = timestamp
            val output = DoubleArray(3) { values[it].toDouble() }
            for (sectionIndex in sections.indices) {
                val c = sections[sectionIndex]
                for (axis in 0..2) {
                    val input = output[axis]
                    val y = c.b0 * input + z1[sectionIndex][axis]
                    z1[sectionIndex][axis] = c.b1 * input - c.a1 * y + z2[sectionIndex][axis]
                    z2[sectionIndex][axis] = c.b2 * input - c.a2 * y
                    output[axis] = y
                }
            }
            return Point(timestamp, output[0].toFloat(), output[1].toFloat(), output[2].toFloat())
        }

        private fun configure(rateHz: Double) {
            sampleRate = rateHz.coerceAtLeast(targetHz)
            val cutoff = min(20.0, min(0.4 * sampleRate, 0.9 * targetHz / 2.0))
            val omega = 2.0 * PI * cutoff / sampleRate
            val c = cos(omega)
            sections = listOf(0.5411961, 1.306563).map { q ->
                val alpha = sin(omega) / (2.0 * q)
                val a0 = 1.0 + alpha
                val b0 = (1.0 - c) / (2.0 * a0)
                Section(b0, (1.0 - c) / a0, b0, -2.0 * c / a0, (1.0 - alpha) / a0)
            }
        }
    }

    companion object { private const val TAG = "SensorCollector" }
}
