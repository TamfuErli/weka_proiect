package Proiektua;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

public class csvCleanToArff {
    public static void main(String[] args) {
        try {
            if (args.length < 3) {
                System.out.println("Uso: java csvCleanToArff <inputCSV> <outputCSV> <outputARFF>");
                return;
            }

            String csvTrain = args[0];       // Archivo CSV de entrada
            String csvTrainMoldatua = args[1]; // Archivo CSV procesado de salida
            String trainFileARFF = args[2];   // Archivo ARFF de salida

            // Llamar al método de conversión
            Instances arffTrain = bihurketaArff(csvTrain, csvTrainMoldatua, trainFileARFF);

            // Opcional: Mostrar información sobre los datos convertidos
            System.out.println("\nDatos convertidos a ARFF:");
            System.out.println(arffTrain.toSummaryString());

        } catch(Exception e) {
            System.err.println("Error en el proceso:");
            e.printStackTrace();
        }
    }

    public static Instances bihurketaArff(String inputFilePath, String outputCSVPath, String outputARFFPath) throws Exception {
        List<String[]> data = new ArrayList<>();
        Set<String> sentimentClasses = new HashSet<>();

        // Leer el archivo CSV original y procesarlo
        try (BufferedReader br = Files.newBufferedReader(Path.of(inputFilePath));
             FileWriter fw = new FileWriter(outputCSVPath);
             BufferedWriter bw = new BufferedWriter(fw)) {

            // Leer encabezado
            String header = br.readLine();
            if (header == null) {
                throw new IOException("El archivo está vacío");
            }

            // Identificar índices de columnas
            String[] headers = header.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1); // Regex para manejar comas dentro de comillas
            int sentimentIndex = -1;
            int tweetTextIndex = -1;

            for (int i = 0; i < headers.length; i++) {
                String headerName = headers[i].trim().replace("\"", "");
                if (headerName.equalsIgnoreCase("Sentiment")) {
                    sentimentIndex = i;
                } else if (headerName.equalsIgnoreCase("TweetText")) {
                    tweetTextIndex = i;
                }
            }

            if (sentimentIndex == -1 || tweetTextIndex == -1) {
                throw new IOException("No se encontraron las columnas 'Sentiment' y/o 'TweetText'");
            }

            // Escribir nuevo encabezado en CSV
            bw.write("sentiment,TweetText\n");

            // Procesar cada línea
            String line;
            while ((line = br.readLine()) != null) {
                String[] columns = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                if (columns.length <= Math.max(sentimentIndex, tweetTextIndex)) continue;

                String sentiment = columns[sentimentIndex].trim().replace("\"", "");
                String tweetText = columns[tweetTextIndex].trim().replace("\"", "")
                        .replaceAll("https?://\\S+", "")    // Quita links
                        .replaceAll("[^a-zA-Z\\s]", "")     // Quita caracteres no alfabéticos
                        .replaceAll("\\s+", " ")            // Normaliza espacios
                        .trim();

                // Escribir en el CSV procesado
                bw.write(String.format("\"%s\",\"%s\"%n", sentiment, tweetText));

                // Almacenar datos para ARFF
                data.add(new String[]{sentiment, tweetText});
                sentimentClasses.add(sentiment);
            }

            System.out.println("Archivo CSV procesado guardado como: " + outputCSVPath);
        } catch (IOException e) {
            throw new Exception("Error procesando el CSV: " + e.getMessage(), e);
        }

        // Crear instancias Weka
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("TweetText", (List<String>) null)); // Atributo de texto
        ArrayList<String> classValues = new ArrayList<>(sentimentClasses);
        attributes.add(new Attribute("sentiment", classValues));

        Instances instances = new Instances("tweets", attributes, data.size());
        instances.setClassIndex(1); // Establecer sentiment como class index

        // Añadir instancias
        for (String[] entry : data) {
            DenseInstance instance = new DenseInstance(2);
            instance.setValue(instances.attribute("TweetText"), entry[1]);
            instance.setValue(instances.attribute("sentiment"), entry[0]);
            instances.add(instance);
        }

        // Guardar ARFF
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(instances);
            saver.setFile(new File(outputARFFPath));
            saver.writeBatch();
            System.out.println("Archivo ARFF guardado como: " + outputARFFPath);
        } catch (Exception e) {
            throw new Exception("Error guardando ARFF: " + e.getMessage(), e);
        }

        return instances;
    }
}