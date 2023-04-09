package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    Button button_Album,button_take_photo,button_real_time_detect;
    ImageView imageview_image_show;
    TextView textView_show_label;
    TextView textView_confidence_level;
    Bitmap bitmap = null;
    Module model = null;
    String classname = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 控件绑定
        button_Album = (Button) findViewById(R.id.button_Album);
        button_take_photo = (Button) findViewById(R.id.button_take_photo);
        button_real_time_detect = (Button) findViewById(R.id.button_real_time_detect);
        imageview_image_show = (ImageView) findViewById(R.id.ImageView_image_show);
        textView_show_label = (TextView) findViewById(R.id.TextView_show_label);
        textView_confidence_level = (TextView) findViewById(R.id.textView_confidence_level);
        //按钮监听
        button_Album.setOnClickListener(this);
        button_take_photo.setOnClickListener(this);
        button_real_time_detect.setOnClickListener(this);
    }
    public void onClick(View v){
        switch (v.getId()){
            case R.id.button_Album:                                                        //打开相册
                Toast.makeText(this,"上传图片检测",Toast.LENGTH_SHORT).show();
                Intent chooseIntent = new Intent(Intent.ACTION_GET_CONTENT);
                chooseIntent.setType("image/*");
                chooseIntent.addCategory(Intent.CATEGORY_OPENABLE);
                startActivityForResult(chooseIntent,001);
                break;
            case R.id.button_take_photo:                                               // 执行拍照检测
                Toast.makeText(this,"拍照检测",Toast.LENGTH_SHORT).show();
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent,002);
                break;
            case R.id.button_real_time_detect:                                      // 开启实时检测界面
                Toast.makeText(this,"实时检测",Toast.LENGTH_SHORT).show();
                Intent realtimeIntent = new Intent(MainActivity.this,Realtime_Detect.class);
                startActivity(realtimeIntent);
                break;
        }
    }
    public void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode,resultCode,data);
        if (requestCode==001){
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),data.getData());
                imageview_image_show.setImageBitmap(bitmap);
                model = LiteModuleLoader.load(MainActivity.assetFilePath(this,"fruit30_pytorch_android_new.ptl"));
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
                final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor(); //前向传播
                final float[] scores = outputTensor.getDataAsFloatArray();  //获取Tensor输出，转化为Java矩阵
                float[] softmax_scores = softmax(scores);  //计算出softmax分数
                float max_score = 0;//分析输出结果(寻找最大值的标签，先初始化变量)
                int max_score_idx = -1;
                for (int i=0;i<softmax_scores.length;i++){  //分析输出结果(开始寻找最大值，获取最大值及其索引)
                    if(softmax_scores[i]>max_score)
                    {
                        max_score = softmax_scores[i];
                        max_score_idx = i;
                    }
                }
                classname = Fruit_Classes.fruitclass_names[max_score_idx]; //通过textview控件显示类别名称
                textView_show_label.setText(classname);
                textView_confidence_level.setText(String.format("%.2f%%",max_score*100));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        else if(requestCode==002){
            try {
                //打开相机
                bitmap = (Bitmap) data.getExtras().get("data");
                imageview_image_show.setImageBitmap(bitmap);
                model = LiteModuleLoader.load(MainActivity.assetFilePath(this,"fruit30_pytorch_android_new.ptl"));
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
                final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor(); //前向传播
                final float[] scores = outputTensor.getDataAsFloatArray();  //获取Tensor输出，转化为Java矩阵
                float[] softmax_scores = softmax(scores);  //计算出softmax分数
                float max_score = 0;//分析输出结果(寻找最大值的标签，先初始化变量)
                int max_score_idx = -1;
                for (int i=0;i<softmax_scores.length;i++){  //分析输出结果(开始寻找最大值，获取最大值及其索引)
                    if(softmax_scores[i]>max_score)
                    {
                        max_score = softmax_scores[i];
                        max_score_idx = i;
                    }
                }
                classname = Fruit_Classes.fruitclass_names[max_score_idx]; //通过textview控件显示类别名称
                textView_show_label.setText(classname);
                textView_confidence_level.setText(String.format("%.2f%%",max_score*100));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
    //获取文件路径
    public static String assetFilePath(Context context,String assetName) throws IOException{
        File file = new File(context.getFilesDir(),assetName);
        if(file.exists() && file.length() > 0){
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)){
            try (OutputStream os = new FileOutputStream(file)){
                byte[] buffer = new byte[4*1024];
                int read;
                while ((read = is.read(buffer)) != -1){
                    os.write(buffer,0,read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
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