package Proiektua;
import java.io.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stopwords.Rainbow;
import weka.core.stopwords.RegExpFromFile;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class BoW {
    public static void main(String[] args) {
        String inputFileARFF = args[0]; // Archivo ARFF de entrada
        String outputFileARFF = args[1]; // Archivo ARFF de salida con Bag of Words

        try {
            // Cargar el archivo ARFF
            DataSource source = new DataSource(inputFileARFF);
            Instances data = source.getDataSet();

            // Configurar la columna de clase (sentiment)

            // Aplicar Bag of Words solo al atributo de texto (asegurarse que no se afecte a la columna de clase)
            StringToWordVector filter = new StringToWordVector();
            filter.setWordsToKeep(5000); // Limitar el número de palabras
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setStopwordsHandler(new Rainbow());
            filter.setAttributeIndices("first-last-1"); // Esto aplica solo a las columnas de texto, no a la clase
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

