package Proiektua;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import java.io.*;
import java.util.ArrayList;

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
            bowData.setClassIndex(bowData.numAttributes()-1);
            System.out.println(bowData.toSummaryString());

            // Paso 3: Selección de características (FSS)
            System.out.println("\n=== PASO 3: Selección de características ===");
            Instances finalData = Fss.applyFSS(bowData, finalARFF, dictionaryFile);

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

        // 1. Guardar la clase original (último atributo)
        data.setClassIndex(data.numAttributes() - 1);
        Attribute classAttr = data.classAttribute();
        ArrayList<String> classValues = new ArrayList<>();
        for (int i = 0; i < classAttr.numValues(); i++) {
            classValues.add(classAttr.value(i));
        }

        // 2. Separar el texto (primer atributo) para BoW
        Remove remove = new Remove();
        remove.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // Weka usa 1-based index
        remove.setInputFormat(data);
        Instances textOnly = Filter.useFilter(data, remove);

        // 3. Aplicar BoW solo al texto
        StringToWordVector bowFilter = new StringToWordVector();
        bowFilter.setInputFormat(textOnly);
        bowFilter.setWordsToKeep(5000);
        bowFilter.setLowerCaseTokens(true);
        bowFilter.setOutputWordCounts(true); // Generar frecuencias numéricas
        Instances bowText = Filter.useFilter(textOnly, bowFilter);

        // 4. Recombinar con la clase original
        Instances finalData = new Instances(bowText);
        Attribute newClassAttr = new Attribute("sentiment", classValues);
        finalData.insertAttributeAt(newClassAttr, finalData.numAttributes());
        finalData.setClassIndex(finalData.numAttributes() - 1);

        // Copiar valores de clase
        for (int i = 0; i < data.numInstances(); i++) {
            finalData.instance(i).setValue(finalData.classIndex(),
                    data.instance(i).classValue());
        }

        // Guardar resultados
        ArffSaver saver = new ArffSaver();
        saver.setInstances(finalData);
        saver.setFile(new File(outputARFF));
        saver.writeBatch();

        return finalData;
    }
}