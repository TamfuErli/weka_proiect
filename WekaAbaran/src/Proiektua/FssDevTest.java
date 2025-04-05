package Proiektua;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import java.io.File;

public class FssDevTest {

    public static void main(String[] args) {
        try {

            String inputArff = args[0];
            String dictionaryFile = args[1];
            String outputArff = args[2];

            // Cargar datos
            Instances rawData = DataSource.read(inputArff);

            // Aplicar FSS con diccionario
            Instances filteredData = applyFssWithDictionary(rawData, dictionaryFile);

            // Guardar resultados
            saveArff(filteredData, outputArff);

        } catch (Exception e) {
            System.err.println("Error durante el procesamiento:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Aplica el filtro FSS usando un diccionario predefinido
     *
     * @param rawData Datos brutos en formato Instances
     * @param dictionaryFile Ruta al archivo de diccionario óptimo
     * @return Instances con los datos filtrados
     * @throws Exception
     */
    public static Instances applyFssWithDictionary(Instances rawData, String dictionaryFile) throws Exception {
        FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();

        // Configurar el filtro
        filter.setDictionaryFile(new File(dictionaryFile));
        filter.setLowerCaseTokens(true);
        filter.setInputFormat(rawData);

        // Aplicar el filtro
        return Filter.useFilter(rawData, filter);
    }

    /**
     * Guarda las instancias en un archivo ARFF
     *
     * @param data Instancias a guardar
     * @param fileName Nombre del archivo de salida
     * @throws Exception
     */
    public static void saveArff(Instances data, String fileName) throws Exception {
        weka.core.converters.ArffSaver saver = new weka.core.converters.ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(fileName));
        saver.writeBatch();
    }
}