package Proiektua;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import java.io.*;

public class aurreProzesamendua {
    public static void main(String[] args) {
        try {
            if (args.length < 5) {
                System.out.println("Uso: java aurreProzesamendua <inputCSV> <cleanCSV> <rawARFF> <bowARFF> <finalARFF> <dictionaryFile>");
                System.out.println("Ejemplo: java aurreProzesamendua tweets.csv tweets_clean.csv tweets_raw.arff tweets_bow.arff tweets_final.arff hiztegia.txt");
                return;
            }

            String inputCSV = args[0];
            String cleanCSV = args[1];
            String rawARFF = args[2];
            String bowARFF = args[3];
            String finalARFF = args[4];
            String dictionaryFile = args[5];

            // Paso 1: Limpiar CSV y convertir a ARFF básico
            System.out.println("\n=== PASO 1: Limpieza de CSV y conversión a ARFF ===");
            Instances rawData = csvCleanToArff.bihurketaArff(inputCSV, cleanCSV, rawARFF);
            System.out.println(rawData.toSummaryString());

            // Paso 2: Aplicar Bag of Words
            System.out.println("\n=== PASO 2: Aplicando Bag of Words ===");
            Instances bowData = applyBagOfWords(rawARFF, bowARFF);
            System.out.println(bowData.toSummaryString());

            // Paso 3: Selección de características (FSS)
            System.out.println("\n=== PASO 3: Selección de características ===");
            Instances finalData = Fss.applyFSS(bowARFF, finalARFF, dictionaryFile);

        } catch (Exception e) {
            System.err.println("Error en el procesamiento:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static Instances applyBagOfWords(String inputARFF, String outputARFF) throws Exception {
        // Cargar datos
        DataSource source = new DataSource(inputARFF);
        Instances data = source.getDataSet();

        // Configurar filtro Bag of Words
        StringToWordVector filter = new StringToWordVector();
        filter.setWordsToKeep(5000);
        filter.setLowerCaseTokens(true);
        filter.setAttributeIndices("first"); // Aplicar solo al primer atributo (texto)
        filter.setInputFormat(data);

        // Aplicar filtro
        Instances newData = Filter.useFilter(data, filter);

        // Guardar resultados
        weka.core.converters.ArffSaver saver = new weka.core.converters.ArffSaver();
        saver.setInstances(newData);
        saver.setFile(new File(outputARFF));
        saver.writeBatch();

        return newData;
    }
}