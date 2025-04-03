package Proiektua;
import java.io.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class BoW {
    public static void main(String[] args) {
        String inputFileARFF = "tweets.arff"; // Archivo ARFF de entrada
        String outputFileARFF = "tweets_bow.arff"; // Archivo ARFF de salida con Bag of Words

        try {
            // Cargar el archivo ARFF
            DataSource source = new DataSource(inputFileARFF);
            Instances data = source.getDataSet();

            // Configurar la columna de clase (sentiment)
            data.setClassIndex(data.numAttributes() - 1);

            // Aplicar Bag of Words con 5000 palabras
            StringToWordVector filter = new StringToWordVector();
            filter.setWordsToKeep(5000);
            filter.setLowerCaseTokens(true);
            filter.setAttributeIndices("first"); // Aplicar al atributo de texto
            filter.setInputFormat(data);

            Instances newData = Filter.useFilter(data, filter);

            // Guardar el nuevo archivo ARFF
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileARFF));
            writer.write(newData.toString());
            writer.close();

            System.out.println("Archivo ARFF con Bag of Words guardado como " + outputFileARFF);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
