package com.example.myapplication;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class Realtime_Detect extends AppCompatActivity {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView camera_view;
    TextView real_time_detect_label;
    TextView real_time_detect_confidence;
    Module model;
    String classname=null;
    float detect_confidence;
    private int REQUEST_CODE_PERMISSION = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permission.CAMERA"};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_realtime_detect);
        camera_view = (PreviewView) findViewById(R.id.camera_view);
        real_time_detect_label = (TextView) findViewById(R.id.realtime_detect_label);
        real_time_detect_confidence = (TextView) findViewById(R.id.realtime_detect_confidence);
        if(!checkPermissions()){
            ActivityCompat.requestPermissions(this,REQUIRED_PERMISSIONS,REQUEST_CODE_PERMISSION);
        }
        LoadTorchModule("fruit30_pytorch_android_new.ptl");
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(()->{
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                start_camera(cameraProvider);
            }catch (ExecutionException | InterruptedException e){
            }
        },ContextCompat.getMainExecutor(this));

    }
    private boolean checkPermissions(){
        for (String permission: REQUIRED_PERMISSIONS){
            if (ContextCompat.checkSelfPermission(this,permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }
    Executor executor = Executors.newSingleThreadExecutor();

    void start_camera(@NonNull ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(camera_view.getSurfaceProvider());
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().setTargetResolution(new Size(224,224)).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                int rotation = image.getImageInfo().getRotationDegrees();
                analyzeImage(image,rotation);
                image.close();
            }
        });
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this,cameraSelector,preview,imageAnalysis);
    }
    void LoadTorchModule(String filename){
        try {
            model = LiteModuleLoader.load(MainActivity.assetFilePath(this,filename));
        }catch (IOException e){
            Log.e("Error","Error reading file",e);
            finish();
        }
    }
    void analyzeImage(ImageProxy image,int rotation){
        @SuppressLint("UnsafeOptInUsageError")Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(image.getImage(), rotation,224,224,TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        float[] softmax_scores = softmax(scores);
        float max_score = 0;
        int max_score_idx = -1;
        for (int i=0;i<softmax_scores.length;i++){
            if(softmax_scores[i]>max_score)
            {
                max_score = softmax_scores[i];
                max_score_idx = i;
            }
        }
        classname = Fruit_Classes.fruitclass_names[max_score_idx];
        detect_confidence = max_score;
        Log.v("Torch","Detected - "+classname);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                real_time_detect_label.setText(classname);
                real_time_detect_confidence.setText(String.format("%.2f%%",detect_confidence*100));
            }
        });
    }
    private static float[] softmax(float[] input) {
        double total = 0;
        for (int i=0;i<input.length;i++){
            total = total+Math.exp(input[i]);
        }
        float[] output = new float[input.length];
        for (int i=0; i<input.length; i++) {
            output[i] = (float) (Math.exp(input[i]) / total);
        }
        return output;
    }
}