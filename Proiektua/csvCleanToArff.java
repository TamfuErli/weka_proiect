package Proiektua;
import java.io.*;
import java.util.*;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

public class csvCleanToArff {
    public static void main(String[] args) {
        String inputFile = "tweetSentimentdev.csv.csv"; // Archivo de entrada
        String outputFileCSV = "cleaned_tweets.csv"; // Archivo CSV de salida
        String outputFileARFF = "tweets.arff"; // Archivo ARFF de salida

        List<String[]> data = new ArrayList<>();
        Set<String> sentimentClasses = new HashSet<>();

        try (BufferedReader br = new BufferedReader(new FileReader(inputFile));
             BufferedWriter bwCSV = new BufferedWriter(new FileWriter(outputFileCSV))) {

            String header = br.readLine(); // Leer encabezado
            if (header == null) return;

            String[] headers = header.split(",");
            int sentimentIndex = -1;
            int tweetTextIndex = -1;

            for (int i = 0; i < headers.length; i++) {
                if (headers[i].trim().equalsIgnoreCase("\"Sentiment\"")) {
                    sentimentIndex = i;
                } else if (headers[i].trim().equalsIgnoreCase("\"TweetText\"")) {
                    tweetTextIndex = i;
                }
            }

            if (sentimentIndex == -1 || tweetTextIndex == -1) {
                System.out.println("No se encontraron las columnas necesarias.");
                return;
            }

            // Escribir el nuevo encabezado en CSV
            bwCSV.write("sentiment,TweetText\n");

            String line;
            while ((line = br.readLine()) != null) {
                String[] columns = line.split(",");
                if (columns.length <= Math.max(sentimentIndex, tweetTextIndex)) continue;

                String sentiment = columns[sentimentIndex].trim();
                String tweetText = columns[tweetTextIndex]
                        .replaceAll("https?://\\S+", "") // Quita links
                        .replaceAll("[^a-zA-Z\s]", "") // Quita caracteres no alfabéticos
                        .trim();

                bwCSV.write(sentiment + "," + tweetText + "\n");
                data.add(new String[]{sentiment, tweetText});
                sentimentClasses.add(sentiment);
            }

            System.out.println("Archivo limpio guardado como " + outputFileCSV);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Convertir CSV a ARFF con Weka
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(outputFileCSV));
            Instances dataInstances = loader.getDataSet();

            // Configurar atributos
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("TweetText", (List<String>) null)); // Atributo de tipo texto
            ArrayList<String> classValues = new ArrayList<>(sentimentClasses);
            attributes.add(new Attribute("sentiment", classValues));

            Instances newData = new Instances("tweets", attributes, data.size());
            newData.setClassIndex(1);

            for (String[] entry : data) {
                DenseInstance instance = new DenseInstance(2);
                instance.setValue(newData.attribute("TweetText"), entry[1]);
                instance.setValue(newData.attribute("sentiment"), entry[0]);
                newData.add(instance);
            }

            ArffSaver saver = new ArffSaver();
            saver.setInstances(newData);
            saver.setFile(new File(outputFileARFF));
            saver.writeBatch();

            System.out.println("Archivo ARFF guardado como " + outputFileARFF);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}