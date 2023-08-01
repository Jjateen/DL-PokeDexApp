package app.ij.mlwithtensorflowlite;
import android.graphics.Bitmap;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.net.Uri;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


import app.ij.mlwithtensorflowlite.ml.ModelPlzWorks;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    Button sel;
    int imageSize = 64;
    private Bitmap img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
//        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        sel = findViewById(R.id.button2);


        sel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch file picker if we have permission
                if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                    Intent filePickerIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(filePickerIntent, 2);
                } else {
                    // Request file read permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 200);
                }
            }
        });

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }

        });
    }

    public void classifyImage(Bitmap image){
        try {
            ModelPlzWorks model = ModelPlzWorks.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 64, 64, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelPlzWorks.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Abra","Aerodactyl","Alakazam","Alolan Sandslash","Arbok","Arcanine","Articuno","Beedrill","Bellsprout","Blastoise","Bulbasaur","Butterfree","Caterpie","Chansey","Charizard","Charmander","Charmeleon","Clefable","Clefairy","Cloyster","Cubone","Dewgong","Diglett","Ditto","Dodrio","Doduo","Dragonair","Dragonite","Dratini","Drowzee","Dugtrio","Eevee","Ekans","Electabuzz","Electrode","Exeggcute","Exeggutor","Farfetchd","Fearow","Flareon","Gastly","Gengar","Geodude","Gloom","Golbat","Goldeen","Golduck","Golem","Graveler","Grimer","Growlithe","Gyarados","Haunter","Hitmonchan","Hitmonlee","Horsea","Hypno","Ivysaur","Jigglypuff","Jolteon","Jynx","Kabuto","Kabutops","Kadabra","Kakuna","Kangaskhan","Kingler","Koffing","Krabby","Lapras","Lickitung","Machamp","Machoke","Machop","Magikarp","Magmar","Magnemite","Magneton","Mankey","Marowak","Meowth","Metapod","Mew","Mewtwo","Moltres","MrMime","Muk","Nidoking","Nidoqueen","Nidorina","Nidorino","Ninetales","Oddish","Omanyte","Omastar","Onix","Paras","Parasect","Persian","Pidgeot","Pidgeotto","Pidgey","Pikachu","Pinsir","Poliwag","Poliwhirl","Poliwrath","Ponyta","Porygon","Primeape","Psyduck","Raichu","Rapidash","Raticate","Rattata","Rhydon","Rhyhorn","Sandshrew","Sandslash","Scyther","Seadra","Seaking","Seel","Shellder","Slowbro","Slowpoke","Snorlax","Spearow","Squirtle","Starmie","Staryu","Tangela","Tauros","Tentacool","Tentacruel","Vaporeon","Venomoth","Venonat","Venusaur","Victreebel","Vileplume","Voltorb","Vulpix","Wartortle","Weedle","Weepinbell","Weezing","Wigglytuff","Zapdos","Zubat"};
            result.setText("\n" + classes[maxPos]);

//            String s = "";
//            for(int i = 0; i < classes.length; i++){
//                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
//            }
//            confidence.setText(s);


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, true);
            classifyImage(image);
        }
        else if (requestCode == 2 && resultCode == RESULT_OK && data != null) {
            Uri selectedImageUri = data.getData();
            try {
                // Get the image bitmap from the selected URI
                Bitmap image = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImageUri);
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, true);
                classifyImage(image);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }
}