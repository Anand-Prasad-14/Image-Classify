package com.example.imageclassify

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.imageclassify.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {

    private lateinit var selectBtn: Button
    private lateinit var predictBtn: Button
    private lateinit var captureBtn: Button
    private lateinit var result: TextView
    private lateinit var imageView: ImageView
    private lateinit var resetBtn: Button
    private lateinit var bitmap: Bitmap

    private lateinit var labels: List<String>

    private val CAMERA_REQUEST_CODE = 101
    private val CAMERA_PERMISSION_CODE = 102
    private val GALLERY_REQUEST_CODE = 100

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.selectBtn)
        predictBtn = findViewById(R.id.predictBtn)
        captureBtn = findViewById(R.id.captureBtn)
        resetBtn = findViewById(R.id.resetBtn)
        result = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)

        labels = loadLabels()
        Log.d("Labels Loaded", "Number of labels loaded: ${labels.size}")

        selectBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, GALLERY_REQUEST_CODE)
        }

        captureBtn.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                openCamera()
            } else {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    CAMERA_PERMISSION_CODE
                )
            }
        }

        resetBtn.setOnClickListener {
            imageView.setImageResource(R.drawable.logo)
            result.text = ""
        }

        predictBtn.setOnClickListener {
            if (::bitmap.isInitialized) {
                try {
                    val model = MobilenetV110224Quant.newInstance(this)

                    // Resize bitmap to 224x224
                    bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

                    // Prepare TensorImage
                    val tensorImage = TensorImage(DataType.UINT8)
                    tensorImage.load(bitmap)

                    // Create input for the model
                    val inputFeature0 =
                        TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
                    inputFeature0.loadBuffer(tensorImage.buffer)

                    // Run model inference
                    val outputs = model.process(inputFeature0)
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer
                    val confidenceArray = outputFeature0.floatArray

                    // Get the predicted label with the highest confidence
                    val maxID = getMax(confidenceArray)
                    if (maxID in labels.indices) {
                        result.text = "Prediction: ${labels[maxID]}"
                    } else {
                        result.text = "Prediction out of bounds"
                        Log.e(
                            "Prediction Error",
                            "maxID: $maxID exceeds label size: ${labels.size}"
                        )
                    }

                    model.close()
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            } else {
                Toast.makeText(this, "Please select or capture an image first", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    private fun loadLabels(): List<String> {
        val labelList = mutableListOf<String>()
        try {
            val bufferedReader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
            var line: String? = bufferedReader.readLine()
            while (line != null) {
                labelList.add(line)
                line = bufferedReader.readLine()
            }
            bufferedReader.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return labelList
    }

    private fun getMax(arr: FloatArray): Int {
        var maxIndex = 0
        for (i in arr.indices) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i
            }
        }
        return maxIndex
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                GALLERY_REQUEST_CODE -> {
                    val selectedImage = data?.data
                    bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImage)
                    imageView.setImageBitmap(bitmap)
                }

                CAMERA_REQUEST_CODE -> {
                    val extras = data?.extras
                    bitmap = extras?.get("data") as Bitmap

                    if (bitmap.config != Bitmap.Config.ARGB_8888) {
                        bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    }
                    imageView.setImageBitmap(bitmap)
                }
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, CAMERA_REQUEST_CODE)
    }
}
