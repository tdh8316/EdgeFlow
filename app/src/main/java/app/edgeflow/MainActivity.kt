package app.edgeflow

import android.annotation.SuppressLint
import android.os.Bundle
import androidx.annotation.Keep
import androidx.appcompat.app.AppCompatActivity
import app.edgeflow.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    companion object {
        // Used to load the 'edgeflow' library on application startup.
        init {
            System.loadLibrary("edgeflow")
        }
    }

    private var modelPath: String = "uninitialized" // The path to the model DAG file
    private var initialized: Boolean = false // C+++ backend initialization status


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Listener for model loading
        binding.loadButton.setOnClickListener {
            modelPath = loadModelDAG()
            if (modelPath.isNotEmpty()) {
                binding.outputText.text.append("[i] Model loaded: $modelPath\n")
            }
        }

        // Listener for EdgeFlow backend initialization
        binding.initButton.setOnClickListener {
            initializeEdgeFlowBackend(modelPath).let { success ->
                if (success) {
                    binding.outputText.text.append("[i] EdgeFlow backend initialized\n")
                    registerJavaCallback(this).let {
                        if (it) {
                            binding.outputText.text.append("[i] Java callback registered\n")
                            initialized = true
                        } else {
                            binding.outputText.text.append("[!] Failed to register Java callback\n")
                        }
                    }
                } else {
                    binding.outputText.text.append("[!] Failed to initialize EdgeFlow backend\n")
                }
            }
        }

        // Listener for inference request
        binding.inferenceButton.setOnClickListener {
            val inputString = binding.inputText.text.toString()
            requestInference(inputString).let { result ->
                if (result) {
                    binding.outputText.text.append("[i] Inference started successfully\n")
                } else {
                    // binding.outputText.text.append("[!] Failed to start inference\n")
                    if (!initialized) {
                        binding.outputText.text.append("[!] EdgeFlow backend is not initialized\n")
                    }
                }
            }
        }

        // Hello world message from C++ backend
        binding.outputText.text.append(stringFromJNI() + "\n");
    }

    /// Load the model DAG from file
    /// @return The path to the model DAG file or an empty string if loading failed
    private fun loadModelDAG(): String {
        // TODO: Load the model DAG

        return "model.json"
    }

    /// Initialize the EdgeFlow backend
    /// @param modelDAGPath The path to the model DAG file
    /// @return True if initialization was successful, false otherwise
    private fun initializeEdgeFlowBackend(
        modelDAGPath: String,
    ): Boolean {
        // TODO: Load device info and device list as JSON format
        val deviceInfoJson = "{}"
        val deviceListJson = "{}"

        return initializeEdgeFlow(modelDAGPath, deviceInfoJson, deviceListJson)
    }

    /// Request inference to the EdgeFlow backend
    /// @param inputString The input string for inference
    /// @return True if inference was started successfully, false otherwise
    private fun requestInference(
        inputString: String,
    ): Boolean {
        if (!initialized) {
            return false
        }

        binding.outputText.text.append("[i] === New inference session ===\n")

        val inputJsonString: String = inputString
        binding.outputText.text.append("[i] Inference input: $inputJsonString\n")
        return startInference(inputString)
    }

    /// Callback from the EdgeFlow backend when inference is complete
    /// @param result The result of the inference
    @SuppressLint("SetTextI18n")
    @Keep
    fun onInferenceComplete(result: String) {
        runOnUiThread {
            val text = binding.outputText.text?.toString() ?: ""
            binding.outputText.setText("$text[i] Inference result: $result\n")
        }
    }

    /* ==========================
     * JNI methods
     * ========================= */
    external fun stringFromJNI(): String

    @Suppress("unused")
    external fun initializeEdgeFlow(
        modelDAGPath: String,
        deviceInfoString: String,
        deviceListString: String,
    ): Boolean

    @Suppress("unused")
    external fun startInference(
        inputString: String,
    ): Boolean

    @Suppress("unused")
    external fun registerJavaCallback(
        thiz: MainActivity,
    ): Boolean
}
